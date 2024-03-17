"""
# Vocabulary and Matching
So far we've seen how a body of text is divided into tokens, and how individual tokens are parsed and tagged with parts of speech, dependencies and lemmas.

Lets understand about identify and label specific phrases that match patterns in NLP

## Rule-based Matching

"""

# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')

# Import the Matcher library
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

##"""<font color=green>Here `matcher` is an object that pairs to the current `Vocab` object. We can add and remove specific named matchers to `matcher` as needed.</font>
##
##### Creating patterns
##In literature, the phrase 'solar power' might appear as one word or two, with or without a hyphen. In this section we'll develop a matcher named 'SolarPower' that finds all three:
##"""
##
##pattern1 = [{'LOWER': 'solarpower'}]
##pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
##pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
##
##matcher.add('SolarPower', [pattern1])
##matcher.add('SolarPower', [pattern2])
##matcher.add('SolarPower', [pattern3])

##"""Let's break this down:
##* `pattern1` looks for a single token whose lowercase text reads 'solarpower'
##* `pattern2` looks for two adjacent tokens that read 'solar' and 'power' in that order
##* `pattern3` looks for three adjacent tokens, with a middle token that can be any punctuation.<font color=green>*</font>
##
##<font color=green>\* Remember that single spaces are not tokenized, so they don't count as punctuation.</font>
##<br>Once we define our patterns, we pass them into `matcher` with the name 'SolarPower', and set *callbacks* to `None` (more on callbacks later).
##
##### Applying the matcher to a Doc object
##"""
##
doc = nlp(u'The Solar Power industry continues to grow as demand \
for solarpower increases. Solar-power cars are gaining popularity.')
##
##found_matches = matcher(doc)
##print(found_matches)
##
"""`matcher` returns a list of tuples. Each tuple contains an ID for the match, with start & end tokens that map to the span `doc[start:end]`"""

##for match_id, start, end in found_matches:
##    string_id = nlp.vocab.strings[match_id]  # get string representation
##    span = doc[start:end]                    # get the matched span
##    print(match_id, string_id, start, end, span.text)
##
##"""The `match_id` is simply the hash value of the `string_ID` 'SolarPower'

##### Setting pattern options and quantifiers
##You can make token rules optional by passing an `'OP':'*'` argument. This lets us streamline our patterns list:
##"""

# Redefine the patterns:
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]

# Remove the old patterns to avoid duplication:
##matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', [pattern1])
matcher.add('SolarPower', [pattern2])

found_matches = matcher(doc)
print(found_matches)

"""This found both two-word patterns, with and without the hyphen!

The following quantifiers can be passed to the `'OP'` key:
<table><tr><th>OP</th><th>Description</th></tr>

<tr ><td><span >\!</span></td><td>Negate the pattern, by requiring it to match exactly 0 times</td></tr>
<tr ><td><span >?</span></td><td>Make the pattern optional, by allowing it to match 0 or 1 times</td></tr>
<tr ><td><span >\+</span></td><td>Require the pattern to match 1 or more times</td></tr>
<tr ><td><span >\*</span></td><td>Allow the pattern to match zero or more times</td></tr>
</table>

### Be careful with lemmas!
If we wanted to match on both 'solar power' and 'solar powered', it might be tempting to look for the *lemma* of 'powered' and expect it to be 'power'. This is not always the case! The lemma of the *adjective* 'powered' is still 'powered':
"""

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}] # CHANGE THIS PATTERN

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', [pattern1])
matcher.add('SolarPower', [pattern2])

doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')

found_matches = matcher(doc2)
print(found_matches)

"""<font color=green>The matcher found the first occurrence because the lemmatizer treated 'Solar-powered' as a verb, but not the second as it considered it an adjective.<br>For this case it may be better to set explicit token patterns.</font>"""

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solarpowered'}]
pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'powered'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', [pattern1])
matcher.add('SolarPower', [pattern2])
matcher.add('SolarPower', [pattern3])
matcher.add('SolarPower', [pattern4])

found_matches = matcher(doc2)
print(found_matches)

##"""## Other token attributes
##Besides lemmas, there are a variety of token attributes we can use to determine matching rules:
##<table><tr><th>Attribute</th><th>Description</th></tr>
##
##<tr ><td><span >`ORTH`</span></td><td>The exact verbatim text of a token</td></tr>
##<tr ><td><span >`LOWER`</span></td><td>The lowercase form of the token text</td></tr>
##<tr ><td><span >`LENGTH`</span></td><td>The length of the token text</td></tr>
##<tr ><td><span >`IS_ALPHA`, `IS_ASCII`, `IS_DIGIT`</span></td><td>Token text consists of alphanumeric characters, ASCII characters, digits</td></tr>
##<tr ><td><span >`IS_LOWER`, `IS_UPPER`, `IS_TITLE`</span></td><td>Token text is in lowercase, uppercase, titlecase</td></tr>
##<tr ><td><span >`IS_PUNCT`, `IS_SPACE`, `IS_STOP`</span></td><td>Token is punctuation, whitespace, stop word</td></tr>
##<tr ><td><span >`LIKE_NUM`, `LIKE_URL`, `LIKE_EMAIL`</span></td><td>Token text resembles a number, URL, email</td></tr>
##<tr ><td><span >`POS`, `TAG`, `DEP`, `LEMMA`, `SHAPE`</span></td><td>The token's simple and extended part-of-speech tag, dependency label, lemma, shape</td></tr>
##<tr ><td><span >`ENT_TYPE`</span></td><td>The token's entity label</td></tr>
##
##</table>
##
##### Token wildcard
##You can pass an empty dictionary `{}` as a wildcard to represent **any token**. For example, you might want to retrieve hashtags without knowing what might follow the `#` character:
##>`[{'ORTH': '#'}, {}]`
##
##___
#### PhraseMatcher
##In the above section we used token patterns to perform rule-based matching. An alternative - and often more efficient - method is to match on terminology lists. In this case we use PhraseMatcher to create a Doc object from a list of phrases, and pass that into `matcher` instead.
##"""
##
### Perform standard imports, reset nlp
##import spacy
##nlp = spacy.load('en_core_web_sm')
##
##from spacy.matcher import PhraseMatcher
##matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
##
##terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
##patterns = [nlp(text) for text in terms]
##matcher.add("TerminologyList", None, *patterns)
##
### Borrowed from https://daringfireball.net/linked/2019/09/21/patel-11-pro
##text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
##               "photography tests pitting the iPhone 11 Pro against the "
##               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 
##matches = matcher(text_doc)
##print(matches)
##
##match_id, start, end = matches[1]
##print(nlp.vocab.strings[match_id], text_doc[start:end])
##
##
