import re
from rdkit import Chem
from typing import Tuple


def canonicalize_smiles(smi: str, map_clear=True, cano_with_heavyatom=True) -> str:
    cano_smi = ''
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        cano_smi = ''
    else:
        if mol.GetNumHeavyAtoms() < 2 and cano_with_heavyatom:
            cano_smi = 'CC'
        elif map_clear:
            for a in mol.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            cano_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return cano_smi


def hydro_remove(subs: str, prod: str) -> Tuple[str, str]:
    #subs = re.sub(r'\[(?P<atom>[a-zA-z])H[0-9]?\]', r'\g<atom>', subs)
    #prod = re.sub(r'\[(?P<atom>[a-zA-Z])H[0-9]?\]', r'\g<atom>', prod)
    subs = re.sub(r'\[(?P<atom>[CNOc])H[0-9]?\]', r'\g<atom>', subs)
    prod = re.sub(r'\[(?P<atom>[CNOc])H[0-9]?\]', r'\g<atom>', prod)

    #atom_index = set([atom for atom in re.findall(r'[a-zA-Z][a-zA-Z]?', subs+'>>'+prod)])
    #symbol_index = set([symbol for symbol in re.findall(r'[^a-zA-Z>>]', subs+'>>'+prod)])

    #return subs, prod, list(atom_index), list(symbol_index)
    return subs, prod


def space_remove(atom_match):
    #print(atom_match)
    return atom_match.group('atom').replace(' ', '')


def atom_space_remove(atom_match):
    match = atom_match.group('atom')
    pattern = r'(?P<atom>[A-z][\s][a-z])'
    match = re.sub(pattern, space_remove, match)
    return match


def space_add(atom_match):
    return ' '.join(atom_match.group('atom'))


def atom_split(subs: str, prod: str) -> Tuple[str, str]:
    plural_letter_atom = (
        r'(?P<atom>H\se|L\si|B\se|N\se|N\sa|M\sg|A\sl|S\si|C\sl|A\sr'
        r'|C\sa|S\sc|T\si|C\sr|M\sn|F\se|C\so|N\si|C\su|Z\sn|G\sa|G\se|A\ss|S\se|B\sr|K\sr|R\sb|S\sr|Z\sr|N\sb|M\so|T\sc|R\su|R\sh|P\sd|A\sg|C\sd|I\sn|S\sn|S\sb|T\se|X\se'
        r'|C\ss|B\sa|L\sa|C\se|P\sr|N\sd|P\sm|S\sm|E\su|G\sd|T\sb|D\sy|H\so|E\sr|T\sm|Y\sb|L\su|H\sf|T\sa|R\se|O\ss|I\sr|P\st|A\su|H\sg|T\si|P\sb|B\si|P\so|A\st|R\sn'
        r'|F\sr|R\sa|A\sc|T\sh|P\sa|N\sp|P\su|A\sm|C\sm|B\sk|C\sf|E\ss|F\sm|M\sd|N\so|L\sr|R\sf|D\sb|S\sg|B\sh|H\ss|M\st|D\ss|R\sg|C\sn|N\sh|F\sl|M\sc|L\sv|T\ss|O\sg)'
    )

    IIIA_to_0 = r'(?P<atom>H\se|N\se|A\sl|S\si|C\sl|A\sr|G\sa|G\se|A\ss|S\se|B\sr|K\sr|I\sn|S\sn|S\sb|T\se|X\se|T\sl|P\sb|B\si|P\so|A\st|R\sn|N\sh|F\sl|M\sc|L\sv|T\ss|O\sg)'

    bracket_string = r'(?P<atom>\[.+?\])'

    subs, prod = ' '.join(subs), ' '.join(prod)
    subs = re.sub(bracket_string, space_remove, subs)
    subs = re.sub(IIIA_to_0, space_remove, subs)
    prod = re.sub(bracket_string, space_remove, prod)
    prod = re.sub(IIIA_to_0, space_remove, prod)
    return subs, prod


def smi2token(smi: str) -> str:
    pattern = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    token_regex = re.compile(pattern)
    tokens = [token for token in token_regex.findall(smi)]
    return " ".join(tokens)


def token2char(smi: str) -> str:
    pattern = r'(?P<atom>\[[^\]]+])'
    atom_pattern = r'(?P<atom>\[[^\]]+[A-Z][\s][a-z][^\]]+\])'
    smi = re.sub(pattern, space_add, smi)
    smi = re.sub(atom_pattern, atom_space_remove, smi)
    return smi


def token_preprocess(subs: str, prod: str) -> Tuple[str, str]:
    #subs, prod = hydro_remove(subs, prod)
    #subs, prod = atom_split(subs, prod)
    subs, prod = smi2token(subs), smi2token(prod)
    return subs, prod


def char_preprocess(subs: str, prod: str) -> Tuple[str, str]:
    subs, prod = smi2token(subs), smi2token(prod)
    subs, prod = token2char(subs), token2char(prod)
    return subs, prod


if __name__ == '__main__':
    smi1 = '[IH:19]'
    smi2 = '[OH2:28]'
    smi1, smi2 = canonicalize_smiles(smi1), canonicalize_smiles(smi2)
    smi1
