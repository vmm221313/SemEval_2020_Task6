

# """Loads dataset and graph if exists, else create and process them from raw data
# Returns --->
# f: torch tensor input of GCN (Identity matrix)
# X: input of GCN (Identity matrix)
# A_hat: transformed adjacency matrix A
# selected: indexes of selected labelled nodes for training
# test_idxs: indexes of not-selected nodes for inference/testing
# labels_selected: labels of selected labelled nodes for training
# labels_not_selected: labels of not-selected labelled nodes for inference/testing
# """

def load_datasets(args, train_test_split=0):

    df_data_path = "df_data.pkl"
    graph_path = "text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph(args.train_data, args.infer_data, args.max_vocab_len)
    df_data = load_pickle("df_data.pkl")
    G_dict = load_pickle("text_graph.pkl")
    G = G_dict["graph"]
    infer_idx_start = G_dict["infer_idx_start"]
    del G_dict
    
    logger.info("Building adjacency and degree matrices...")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    
    if train_test_split == 1:
        logger.info("Splitting labels for training and inferring...")
        ### stratified test samples
        test_idxs = []
        for b_id in df_data["label"].unique():
            dum = df_data[df_data["label"] == b_id]
            if len(dum) >= 4:
                test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))
        save_as_pickle("test_idxs.pkl", test_idxs)
        # select only certain labelled nodes for semi-supervised GCN
        selected = []
        for i in range(len(df_data)):
            if i not in test_idxs:
                selected.append(i)
        save_as_pickle("selected.pkl", selected)
    else:
        logger.info("Preparing training labels...")
        test_idxs = [i for i in range(infer_idx_start, len(df_data))]
        selected = [i for i in range(infer_idx_start)]
        save_as_pickle("selected.pkl", selected)
        save_as_pickle("test_idxs.pkl", test_idxs)
    
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_selected = list(df_data.loc[selected]['label'])
    if train_test_split == 1:    
        labels_not_selected = list(df_data.loc[test_idxs]['label'])
    else:
        labels_not_selected = []
        
    f = torch.from_numpy(f).float()
    save_as_pickle("labels_selected.pkl", labels_selected)
    save_as_pickle("labels_not_selected.pkl", labels_not_selected)
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs
