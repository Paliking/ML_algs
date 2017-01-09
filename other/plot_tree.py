from sklearn import tree
    i_tree = 0
    for tree_in_forest in regressor.estimators_:
        with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
            my_file = tree.export_graphviz(tree_in_forest, feature_names=features_namelist, out_file = my_file)
        i_tree = i_tree + 1
        if i_tree == 5:
            break