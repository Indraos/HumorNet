# !pip install phyme
from Phyme import Phyme
import itertools

ph = Phyme()
word = 'friend'
# find perfect rhymes. DOG -> COG
def get_rhymes(word):
  """
  Return list of all rhymes given word
  """
  return list(itertools.chain.from_iterable(ph.get_perfect_rhymes(word ).values()))

print(ph.get_perfect_rhymes(word))
print(get_rhymes(word))

# # Other functionality:

# def get_rhymes(word):
#   rhymes = []
#   for rhyming_list in 

# # find rhymes with the same vowels and consonants of the same type (fricative, plosive, etc) and voicing (voiced or unvoiced). FOB -> DOG
# ph.get_family_rhymes(word, num_sylls=None)

#  # find rhymes with the same vowels and consonants of the same type, regardless of voicing. HAWK -> DOG
# ph.get_partner_rhymes(word)

# # find rhymes with the same vowels and consonants, as well as any extra consonants. DUDES -> DUES
# ph.get_additive_rhymes(word)

# # find rhymes with the same vowels and a subset of the same consonants. DUDE -> DO
# ph.get_subtractive_rhymes(word)  

# # find rhymes with the same vowels and some of the same consonants, with some swapped out for other consonants. FACTOR -> FASTER
# ph.get_substitution_rhymes(word) 

# # find rhymes with the same vowels and arbitrary consonants. CASH -> CATS
# ph.get_assonance_rhymes(word)

# # find word that do not have the same vowels, but have the same consonants. CAT -> BOT
# ph.get_consonant_rhymes(word)
