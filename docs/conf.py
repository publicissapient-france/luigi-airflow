my_variables_selection = ["iid", "pid", "match", "date", "go_out", "sports", "tvsports", "exercise",
                          "dining",
                          "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                          "movies",
                          "concerts", "music", "shopping", "yoga"]

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

output_dir = "/Users/sandrapietrowska/Documents/Trainings/luigi/luigi-airflow/output"
best_parameters_file_path = "%s/best_parameters.json" % output_dir
path_eval = "%s/eval.csv" % output_dir
feature_engineered_dataset_file_path = "%s/feature_engineered_dataset.csv" % output_dir
processed_features_names_file_path = "%s/processed_features_names.csv" % output_dir
workspace="/Users/sandrapietrowska/Documents/Trainings/luigi/data_source/"
