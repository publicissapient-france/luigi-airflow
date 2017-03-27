my_variables_selection = ["iid", "pid", "match", "gender", "date", "go_out", "sports", "tvsports", "exercise",
                          "dining",
                          "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater",
                          "movies",
                          "concerts", "music", "shopping", "yoga"]

# TODO make features externalized
features = list(["gender", "date", "go_out", "sports", "tvsports", "exercise", "dining", "museums", "art",
                 "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music",
                 "shopping", "yoga"])

parameters = [
    {'max_depth': [8, 10, 12, 14, 16, 18, 20, 30],
     'min_samples_split': [10, 15, 20, 25, 30],
     'min_samples_leaf': [10, 15, 20, 25, 30]
     }
]
scores = ['precision', 'recall']

output_dir = "/home/dolounet/dev/workshops/luigi-airflow/output"
best_parameters_file_path = "%s/best_parameters.json" % output_dir
feature_engineered_dataset_file_path = "%s/feature_engineered_dataset.csv" % output_dir
workspace="/home/dolounet/dev/workshops/"
