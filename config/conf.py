my_variables_selection = ["iid", "pid", "match", "date", "go_out", "sports", "tvsports", "exercise",
                          "dining",
                          "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                          "movies",
                          "concerts", "music", "shopping", "yoga"]

sport_variables = ["sports", "exercise", "hiking", "yoga"]

my_variables_selection_pred = ["iid", "pid", "date", "go_out", "sports", "tvsports", "exercise",
                          "dining",
                          "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                          "movies",
                          "concerts", "music", "shopping", "yoga"]

# TODO make features externalized
features = list(["date", "go_out", "sports", "tvsports", "exercise", "dining", "museums", "art",
                 "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music",
                 "shopping", "yoga"])

parameters = [
    {'max_depth': [8, 10],
     'min_samples_split': [10, 15],
     'min_samples_leaf': [10, 15]
     }
]

scores = ['precision', 'recall']

#workspace = "/home/dolounet/dev/workshops/luigi-airflow-hands-on/"
workspace = "/usr/local/love_matcher_project/"
output_dir = workspace + "output/"
best_parameters_file_path = "%s/best_parameters.json" % output_dir
path_eval = "%s/eval.csv" % output_dir
feature_engineered_dataset_file_path = "%s/feature_engineered_dataset.csv" % output_dir
processed_features_names_file_path = "%s/processed_features_names.csv" % output_dir
data_source = workspace + "data_source/"
