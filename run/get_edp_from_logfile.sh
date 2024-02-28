
paste <(grep -A 2 "Evaluation results:" saved_logs/ours___2024.02.19-08.47.19.673/ours___2024.02.19-08.47.19.673.log | awk -F"=" '{if ($0 ~ /Latency/) print }') <(grep -A 2 "Evaluation results:" saved_logs/ours___2024.02.19-08.47.19.673/ours___2024.02.19-08.47.19.673.log | awk -F"=" '{if ($0 ~ /Energy/) print }') | awk '{print "Latency=", $1, "Energy=", $2, "EDP=", $1*$2}'
