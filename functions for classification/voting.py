from collections import Counter


def majority_voting(predictions):

    # Count the occurrences of each prediction
    counter = Counter(predictions)

    # Get the most common class
    most_common_class, _ = counter.most_common(1)[0]
    sec_common_class, _ = counter.most_common(2)[0]
    if len(counter.most_common())>2:
        if counter.most_common()[0][1] == counter.most_common()[1][1]:
            if (counter.most_common()[0][0] == predictions[0]) or (counter.most_common()[1][0] == predictions[0]):
                return predictions[0]

    return most_common_class
