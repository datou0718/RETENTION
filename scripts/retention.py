import numpy as np

class RetentionModel:
    
    def __init__(self, model, model_type, cam_size, strategy):
        # RandomForestClassifier from sklearn or XGBClassifier from xgboost
        self.model = model
        # 'RF' for Random Forest, 'XGB' for XGBoost
        self.model_type = model_type
        # e.g., self.cam_size = 64 means 64x64 CAMs
        self.cam_size = cam_size
        # strategy for data placement
        # ODR: occurence-based path reordering
        # SPC: similarity-based path clustering
        # NUM: naive unified mapping
        # NIM: naive independent mapping
        self.strategy = strategy
        # list of list of paths, each path is a list of (feature_index, threshold)
        # for 'NIM', each tree has its own pool of paths
        # for other strategies, all paths are combined into a single pool
        self.pools = []
        # mapping from feature name to index (for XGBoost)
        self.feature_map = {}

    def extract_paths(self):
        
        assert self.model_type in ['RF', 'XGB'], "Invalid model type"

        pool_per_tree = []
        if self.model_type == 'RF':
            for estimator in self.model.estimators_:
                paths = self.extract_paths_from_estimator(estimator)
                pool_per_tree.append(paths)
        else:
            booster = self.model.get_booster()
            df = booster.trees_to_dataframe()
            
            # create a mapping from feature name to index
            feat_names = booster.feature_names
            if feat_names is None:
                feat_names = sorted({f for f in df["Feature"].unique() if f != "Leaf"})
            self.feature_map = {name: i + 1 for i, name in enumerate(feat_names)}

            for tree_id, tdf in df.groupby('Tree'):
                paths = self.extract_paths_from_tdf(tree_id, tdf)
                pool_per_tree.append(paths)

        if self.strategy == 'NIM':
            self.pools = pool_per_tree
        else:
            all_paths = []
            for pool in pool_per_tree:
                all_paths.extend(pool)
            self.pools = [all_paths]

        return
    
    # Extract path helper for RandomForestClassifier
    def extract_paths_from_estimator(self, estimator):

        tree = estimator.tree_
        feature = tree.feature
        threshold = tree.threshold
        left = tree.children_left
        right = tree.children_right

        def is_leaf(nid: int) -> bool:
            return feature[nid] < 0 or left[nid] == right[nid]

        paths = []
        stack = [(0, [])] 
        
        while stack:
        
            nid, cur = stack.pop()
            if is_leaf(nid):
                paths.append(list(set(cur)))
                continue

            fidx = int(feature[nid])
            thr = float(threshold[nid])
            nxt = cur + [(fidx, thr)]

            if right[nid] != -1:
                stack.append((right[nid], nxt))
            if left[nid] != -1:
                stack.append((left[nid], nxt))

        return paths

    # Extract path helper for XGBClassifier
    def extract_paths_from_tdf(self, tree_id, tdf):
        
        nodes = {}

        for _, row in tdf.iterrows():
            nid = row['ID']
            feat_raw = row['Feature']

            if feat_raw == 'Leaf':
                feat = -1
            else:
                feat = self.feature_map.get(feat_raw)

            split = None
            if feat != -1:
                split = float(row['Split'])

            nodes[nid] = {
                'feature': feat,
                'split': split,
                'left': row['Yes'],
                'right': row['No'],
            }

        root = None
        root_id = f"{tree_id}-0"
        root_rows = tdf[tdf['ID'] == root_id]
        root = root_rows.iloc[0]['ID']
        paths = []

        def dfs(nid, cur):
            n = nodes.get(nid)
            if n['feature'] == -1:
                paths.append(list(set(cur))) 
                return

            cond = (n['feature'], n['split'])
            nxt = cur + [cond]

            if n['left'] is not None:
                dfs(n['left'], nxt)
            if n['right'] is not None:
                dfs(n['right'], nxt)

        if root is not None and nodes:
            dfs(root, [])

        return paths

    # Extract unique conditions from paths with their occurences
    def extract_conditions(self, paths):
        conditions = {}
        for path in paths:
            for condition in path:
                if condition in conditions:
                    conditions[condition] += 1
                else:
                    conditions[condition] = 1
        return conditions

    # Occurence-based Path Reordering
    def strategy_odr(self):
        
        conditions = self.extract_conditions(self.pools[0])
        sorted_conditions = sorted(conditions.items(), key=lambda x: x[1], reverse=True)
        
        # initialize cam with 0 (0 = not used, 1 = used)
        rows = int(np.ceil(len(self.pools[0]) / self.cam_size) * self.cam_size)
        cols = int(np.ceil(len(sorted_conditions) / self.cam_size) * self.cam_size)
        cam = np.zeros((rows, cols), dtype=int)
        row_index = 0
        mapped_paths = set()

        # map paths to CAMs 
        for condition_index in range(len(sorted_conditions)-1, -1, -1):
            condition = sorted_conditions[condition_index][0]
            for path_index, path in enumerate(self.pools[0]):
                if path_index in mapped_paths:
                    continue
                if condition in path:
                    for cond_idx in range(len(sorted_conditions)):
                        if sorted_conditions[cond_idx][0] in path:
                            cam[row_index][cond_idx] = 1
                    mapped_paths.add(path_index)
                    row_index += 1
            if row_index >= len(self.pools[0]):
                break

        # count the number of used CAMs
        height = int(np.ceil(cam.shape[0]/self.cam_size))
        width = int(np.ceil(cam.shape[1]/self.cam_size))
        valid = np.zeros((height, width), dtype=bool)
        
        # if any cell in the CAM subarray is used, the subarray is required
        for i in range(height):
            for j in range(width):
                valid[i][j] = np.sum(cam[i*self.cam_size:(i+1)*self.cam_size, j*self.cam_size:(j+1)*self.cam_size]) > 0

        return np.sum(valid)

    # Similarity-based Path Clustering
    def strategy_spc(self):
        
        max_path_len = max([len(path) for path in self.pools[0]])
        assert max_path_len <= self.cam_size, "CAM size must be capable to hold the longest path"

        num_of_clusters = 0
        current_cluster_size = 0
        current_cluster = set()
        mapped_paths = set()

        while len(mapped_paths) < len(self.pools[0]):

            # SPC would be slow for large number of paths
            # uncomment the following lines to see the progress
            # if len(mapped_paths) % 100 == 0:
            #     print(f"Mapped {len(mapped_paths)} paths out of {len(self.pools[0])}")

            max_similarity = -1
            max_path_index = -1
            min_union_len = self.cam_size

            for path_index, path in enumerate(self.pools[0]):
                
                if path_index in mapped_paths:
                    continue
                
                similarity = len(current_cluster.intersection(set(path)))
                union_len = len(current_cluster.union(set(path)))
                
                if similarity > max_similarity and union_len <= self.cam_size or similarity == max_similarity and union_len <= min_union_len:
                    max_similarity = similarity
                    max_path_index = path_index
                    min_union_len = union_len

            if max_path_index == -1:
                num_of_clusters += 1
                current_cluster_size = 0
                current_cluster = set()
            else:
                # add the most similar path to the current cluster
                current_cluster.update(set(self.pools[0][max_path_index]))
                current_cluster_size += 1
                mapped_paths.add(max_path_index)

            if current_cluster_size == self.cam_size:
                num_of_clusters += 1
                current_cluster_size = 0
                current_cluster = set()

        if current_cluster_size > 0:
            num_of_clusters += 1

        return num_of_clusters

    # Naive Unified Mapping
    def strategy_num(self, paths):
        
        conditions = self.extract_conditions(paths)
        x = int(np.ceil(len(conditions) / (self.cam_size)))
        y = int(np.ceil(len(paths) / (self.cam_size)))

        return x * y

    # Naive Independent Mapping
    def strategy_nim(self):
        
        number_of_cams = 0
        
        for pool in self.pools:
            number_of_cams += self.strategy_num(pool)
        
        return number_of_cams

    def data_placement_optimization(self):

        assert self.strategy in ['ODR', 'SPC', 'NUM', 'NIM'], "Invalid strategy"
        num_cams = 0
        
        if self.strategy == 'ODR':
            num_cams = self.strategy_odr()
        elif self.strategy == 'SPC':
            num_cams = self.strategy_spc()
        elif self.strategy == 'NUM':
            num_cams = self.strategy_num(self.pools[0])
        elif self.strategy == 'NIM':
            num_cams = self.strategy_nim()
        
        return num_cams
    
    def run_retention(self):
        
        self.extract_paths()
        cam = self.data_placement_optimization()
        print(f"Total CAMs required for {self.strategy}: {cam}")
        
        return
