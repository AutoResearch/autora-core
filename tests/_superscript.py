"""Functions to convert from normal to superscript characters.
After https://stackoverflow.com/questions/8651361/how-do-you-print-superscript/58612677#58612677
with thanks to norok2.


Examples:
    >>> "".translate(to_superscript)
    ''
    >>> "0".translate(to_superscript)
    '⁰'
    >>> "the quick brown fox jumps over the lazy dog".translate(to_superscript)
    'ᵗʰᵉ ۹ᵘᶦᶜᵏ ᵇʳᵒʷⁿ ᶠᵒˣ ʲᵘᵐᵖˢ ᵒᵛᵉʳ ᵗʰᵉ ˡᵃᶻʸ ᵈᵒᵍ'
    >>> "CCV (FTW)".translate(to_superscript)
    'ᶜᶜⱽ ⁽ᶠᵀᵂ⁾'
"""

superscript_map = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ᶦ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "q": "۹",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    "A": "ᴬ",
    "B": "ᴮ",
    "C": "ᶜ",
    "D": "ᴰ",
    "E": "ᴱ",
    "F": "ᶠ",
    "G": "ᴳ",
    "H": "ᴴ",
    "I": "ᴵ",
    "J": "ᴶ",
    "K": "ᴷ",
    "L": "ᴸ",
    "M": "ᴹ",
    "N": "ᴺ",
    "O": "ᴼ",
    "P": "ᴾ",
    "Q": "Q",
    "R": "ᴿ",
    "S": "ˢ",
    "T": "ᵀ",
    "U": "ᵁ",
    "V": "ⱽ",
    "W": "ᵂ",
    "X": "ˣ",
    "Y": "ʸ",
    "Z": "ᶻ",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
}
to_superscript = str.maketrans(superscript_map)
