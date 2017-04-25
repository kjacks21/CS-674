def print_top_words(model, feature_names, n_top_words, output_file):
    for topic_idx, topic in enumerate(model.components_):
        line1 = "Topic #%d:\n" % topic_idx
        line2 = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(line1)
        print(line2)
        output_file.write(line1)
        output_file.write(line2+"\n")
    print()

