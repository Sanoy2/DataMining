import confusion_displayer as cd

def get_points(confusion): # more is better
    percent_of_not_spam_classified_as_spam = confusion[0][1]/confusion[0][0]
    if(percent_of_not_spam_classified_as_spam < 0.01):
        result =  -confusion[1][0]/confusion[1][1]
    else:
        result = -1000000

    # print("Result: {0}".format(result))
    # print("Percent of critical classification: {0}".format(percent_of_not_spam_classified_as_spam))
    return result


# def get_points_alt(confusion): # more is better
#     spam_as_not_spam = -5
#     spam_as_spam = 10
#     not_spam_as_not_spam = 5
#     percent_of_not_spam_classified_as_spam = confusion[0][1]/confusion[0][0] * 100
#     if(percent_of_not_spam_classified_as_spam < 0.01):
#         not_spam_as_spam = -5
#     else:
#         not_spam_as_spam = -1000

#     result = 0
#     result = result + confusion[0][0] * not_spam_as_not_spam
#     result = result + confusion[0][1] * not_spam_as_spam
#     result = result + confusion[1][0] * spam_as_not_spam
#     result = result + confusion[1][1] * spam_as_spam
#     return result