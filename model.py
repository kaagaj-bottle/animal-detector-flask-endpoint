from load_model import vision_model_constructor, audio_model_constructor
import torch

vision_model = vision_model_constructor()
audio_model = audio_model_constructor()
vision_classes=['asian_elephant', 'asiatic_rock_python', 'barking_deer', 'barn_owl', 'bengal_monitor', 'besra', 'black_baza', 'black_buck', 'black_giant_squirrel', 'brown_bear', 'brown_big-eared_bat', 'burmese_python', 'chital', 'common_noctule', 'egyptian_vulture', 'eurasian_lynx', 'eurasian_otter', 'five_striped_palm_squirrel', 'gaur', 'golden_eagle', 'golden_jackal', 'gray_wolf', 'great_himalayan_leaf_nosed_bat', 'great_hornbill', 'greater_horseshoe_bat', 'greater_one-horned_rhino', 'greater_short_nosed_fruit_bat', 'himalayan_striped_squirrel', 'honey_badger', 'indian_crested_porcupine', 'indian_flying_fox', 'indian_grey_mongoose', 'indian_hare', 'king_cobra', 'knob_billed_duck', 'leopard', 'malayan_porcupine', 'marsh_crocodile', 'masked_palm_civet', 'nepal_gray_langur', 'nilgai', 'oriental_rat_snake', 'peregrine_falcon', 'red_breasted_parakeet', 'red_fox', 'rhesus_macaque', 'royal_bengal_tiger', 'sambar_deer', 'sarus_crane', 'serotine_bat', 'siberian_weasel', 'spectacled_cobra', 'western_marsh_harrier', 'white_tailed_eagle', 'wild_boar']

audio_classes=['asian_koel', 'barn_owl', 'black_kite', 'common_cuckoo', 'common_pochard', 'common_wood_piegon', 'golden_eagle', 'great_hornbill', 'great_slaty_woodpecker', 'greater_spotted_eagle', 'grey_treepie', 'house_crow', 'house_sparrow', 'large-billed_crow', 'long-tailed_duck', 'peregrine_falcon', 'red-billed_blue_magpie', 'rose-ringed_parakee', 'rufous_treepie', 'rustic_bunting', 'sarus_crane', 'spotted_dove', 'western_marsh_harrier']

# check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


def vision_output(input):
    vision_model.eval()
    with torch.no_grad():
        return vision_model(input)


def audio_output(input):
    audio_model.eval()
    with torch.no_grad():
        return audio_model(input)
