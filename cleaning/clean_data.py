import csv
import re


# marks which positions in s
# are utf bytes i.e. \x01\x02\x03
# replaces them with white space without messing up length
def mark_utf_bytes(s):
    form = re.compile("\\\\x..\\\\x..\\\\x..")
    return form.sub("            ", s)


# Look in review for quotes referring to parts of text
# Replaces said quotes with "[QUOT]" token
def place_quote_tokens(passage, review):
    # Users retype quotes so following could vary:
    # - Capitalization
    # - Punctuation (commas, periods)
    # Fix by removing these from both, but keep positions
    # in original strings to map back afterwards
    exclude_chars = [",", ";", ":", ".", "'"]
    pass_len = len(passage)

    def shrink(s):
        shrink_s = ""
        shrink_s_inds = []
        s = mark_utf_bytes(s)
        for i, char in enumerate(s):
            if i == 0 or i == 1 or i == pass_len:
                continue  # Ignore the b' '
            if not char.isalnum() and char != " ":
                continue
            else:
                shrink_s += char
                shrink_s_inds.append(i)
        shrink_s = shrink_s.lower()
        return shrink_s, shrink_s_inds

    shrink_passage, _ = shrink(passage)
    shrink_review, sr_inds = shrink(review)
    # No way to be sure if substring is a "quote" or just reuseing words
    # Simplest solution: consider substrings above a min length quotes

    def remove_empty(L):
        return list(filter(lambda s: s != "", L))

    min_quote_length = 4  # min length (word count)
    passage_words = remove_empty(shrink_passage.split(" "))
    review_words = remove_empty(shrink_review.split(" "))
    if len(review_words) < min_quote_length:
        return review  # No room for a quote

    # All 5 word substrings
    review_subs = [
        " ".join(review_words[i : i + min_quote_length])
        for i in range(len(review_words) - min_quote_length)
    ]
    is_quote = [False for _ in review_subs]
    no_matches = True

    for i, sub in enumerate(review_subs):
        if shrink_passage.find(sub) != -1:
            is_quote[i] = True
            no_matches = False

    if no_matches:
        return review

    # For every sub that is quote, find its start index in word list
    quote_start_inds = []
    for i, _ in enumerate(review_subs):
        if is_quote[i]:
            quote_start_inds.append(i)

    # Algorithm to merge overlapping quote intervals
    # Assumes start times are sorted, all intervals are fixed length
    def merge_intervals(starts, l=min_quote_length):
        intervals = [[start, start + l - 1] for start in starts]
        stack = [intervals.pop(0)]
        while intervals:
            start, end = intervals[0]
            if start <= stack[0][1]:
                stack[0][1] = end
                intervals.pop(0)
            else:
                stack.insert(0, intervals.pop(0))
        stack.reverse()
        return stack

    quote_word_intervals = merge_intervals(quote_start_inds)
    # Need to convert back to intervals in original review now

    # First get char positions of all words
    start_space = shrink_review[0] == " "
    word_inds = [0] if not start_space else []
    end_ind = len(shrink_review) - 1
    for i, char in enumerate(shrink_review):
        if char == " ":
            if i == end_ind:
                break
            if shrink_review[i + 1] != " ":
                word_inds.append(i + 1)
    # Convert intervals to be over chars in shrink_review
    quote_char_intervals = []

    for interval in quote_word_intervals:
        start, end = interval
        char_start = word_inds[start]
        char_end = word_inds[end]
        char_end += len(review_words[end]) - 1
        quote_char_intervals.append((char_start, char_end))

    # Convert these intervals to be in terms of original review string
    quote_intervals = []
    for _, interval in enumerate(quote_char_intervals):
        start, end = interval
        quote_intervals.append((sr_inds[start], sr_inds[end]))

    # Now replace with quote tokens
    # Have to do in reverse order
    quote_intervals.reverse()

    def replace(s, start, end, new):
        return s[:start] + new + s[end:]

    for (start, end) in quote_intervals:
        review = replace(review, start, end + 1, "[QUOT]")

    return review


PATH_IN = "'dataset.csv'"

reader = csv.reader(open(PATH_IN, "r"))
writer = csv.writer(open("cc_quotes_cleaned.csv", "w"))

row_count = 0
features = next(reader)
writer.writerow(["story_target", "target_comment"])

for row in reader:
    if row_count > 0:
        passage, review = row[7], row[8]
        new_review = place_quote_tokens(passage, review)
        writer.writerow([passage, new_review])

    row_count += 1
