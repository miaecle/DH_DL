# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:29:10 2020

@author: Zhenqin Wu
"""

"""
df = pd.read_table('data/plateOntogenyDatasets/plateOntogenyTable.txt')
datasets = set(df['Dataset'])
phenotypes = set(zip(df['Dataset'], df['Phenotype'], df['Ontogenetic_order'], df['CombinedMouseAndHumanOrder'], df['Branching']))
for dataset in datasets:
    phens = [p for p in phenotypes if p[0] == dataset]
    print("ALL_LABELS[\"%s\"] = {" % dataset)
    for phe in phens:
        print("\t\"%s\": %s," % (phe[1], str((phe[-1], phe[-2]))))
    print("}")
""" 

ALL_LABELS = {}

ALL_LABELS["Preimplant_mouse_embryo_Tang_et_al"] = {
	"Zygote": (('Z1',), 1),
	"2_cell": (('Z1', 'Z2'), 2),
	"4_cell": (('Z1', 'Z2', 'Z3'), 3),
	"8_cell": (('Z1', 'Z2', 'Z3', 'Z4'), 4),
	"16_cell": (('Z1', 'Z2', 'Z3', 'Z4', 'Z5'), 5),
	"earlyblast": (('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6'), 6),
	"midblast": (('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7'), 7),
	"lateblast": (('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8'), 8),
}


ALL_LABELS["AT2_AT1_lineage_C1"] = {
	"Embryonic bipotent alveolar progenitor ": (('Av1',), 16),
	"Type II pneumocyte ": (('Av1', 'Av2'), 18),
	"Type I pneumocyte ": (('Av1', 'Av2', 'Av3'), 20),
}


ALL_LABELS["Dendritic_cells_C1"] = {
	"Monocyte-dendritic cell progenitor ": (('O', 'O1', 'MD1'), 17),
	"Common dendritic progenitor ": (('O', 'O1', 'MD1', 'D1'), 18),
	"Pre-dendritic cell ": (('O', 'O1', 'MD1', 'D1', 'D2'), 20),
}


ALL_LABELS["HSPCs_C1"] = {
	"Hematopoietic stem cell progenitor (HSCP) ": (('O',), 14),
	"Multi-lineage primed hematopoietic progenitor ": (('O', 'O1'), 15),
	"Monocyte-dendritic cell progenitor ": (('O', 'O1', 'MD1'), 17),
	"Monocyte progenitor ": (('O', 'O1', 'MD1', 'M1'), 20),
	"Myelocyte ": (('O', 'O1', 'G1'), 19),
	"Granulocyte progenitor ": (('O', 'O1', 'G1', 'G2'), 20),
	"Megakaryocyte progenitor": (('O', 'O1', 'Mk1'), 20),
	"Erythroid progenitor ": (('O', 'O1', 'E1'), 20),
}


ALL_LABELS["Bone_marrow_Smartseq2"] = {
	"Hematopoietic stem and progenitors (KLS) ": (('O', 'O1'), 15),
    "Monocyte progenitor and monocytes ": (('O', 'O1', 'MD1', 'M1'), 20),
	"Early immature B cell ": (('O', 'O1', 'B1'), 20),
	"Late immature B cell ": (('O', 'O1', 'B1', 'B2'), 21),
	"Mature B cell ": (('O', 'O1', 'B1', 'B2', 'B3'), 22),
	"Early immature (proliferating) granulocytes ": (('O', 'O1', 'G1', 'G2', 'G3'), 21),
	"Mature (resting) granulocytes": (('O', 'O1', 'G1', 'G2', 'G3', 'G4'), 22),
	"Late immature granulocytes ": (('O', 'O1', 'G1', 'G2', 'G3', 'G4'), 22),
}


ALL_LABELS["HumanEmbryo_CountTable_New.rds"] = {
    'h2_cell': (('HZ1', 'HZ2'), 2),
    'h4_cellembryo': (('HZ1', 'HZ2', 'HZ3'), 3),
    'h8_cellembryo': (('HZ1', 'HZ2', 'HZ3', 'HZ4'), 4),
    'hMorulae': (('HZ1', 'HZ2', 'HZ3', 'HZ4', 'HZ5'), 5),
    'hZygote': (('HZ1',), 1),
}


ALL_LABELS["Kyle_CountTable_New.rds"] = {
    'APS': (('Kyle', 'hESC', 'APS'), 9),
    'Drmmtm': (('Kyle', 'hESC', 'APS', 'PXM', 'Smtmrs', 'ESMT', 'Drmmtm'), 13),
    'ESMT': (('Kyle', 'hESC', 'APS', 'PXM', 'Smtmrs', 'ESMT'), 12),
    'PXM': (('Kyle', 'hESC', 'APS', 'PXM'), 10),
    'Sclrtm': (('Kyle', 'hESC', 'APS', 'PXM', 'Smtmrs', 'ESMT', 'Sclrtm'), 13),
    'Smtmrs': (('Kyle', 'hESC', 'APS', 'PXM', 'Smtmrs'), 11),
    'hESC': (('Kyle', 'hESC'), 6),
}


"""
df = pd.read_table('data/dropletOntogenyDatasets/dropletOntogenyTable.txt')
datasets = set(df['Dataset'])
phenotypes = set(zip(df['Dataset'], df['Phenotype'], df['Ontogenetic_order'], df['CombinedMouseAndHumanOrder']))
valid_phenotypes = [phe for phe in phenotypes if phe[-1] == phe[-1]]
for dataset in datasets:
    phens = [p for p in valid_phenotypes if p[0] == dataset]
    print("ALL_LABELS[\"%s\"] = {" % dataset)
    for phe in phens:
        print("\t\"%s\": %s," % (phe[1], str((int(phe[-1]),))))
    print("}")
"""

ALL_LABELS["Mouse_Data_StandardProtocol_Neuron.rds"] = {
	"ESC": (('N0',), 6),
	"NP": (('N0', 'N1'), 13),
	"MNP": (('N0', 'N1', 'N2'), 20),
}

ALL_LABELS["Mouse_Data_DirectProtocol_Neuron.rds"] = {
	"ESC": (('N0',), 6),
	"NP": (('N0', 'N1'), 13),
}

ALL_LABELS["Mouse_Data_Marrow_10x_MACA.rds"] = {
	"Stem_Progenitors": (('O', 'O1'), 15),
	"Immature_B": (('O', 'O1', 'B1'), 20),
	"Mature_B": (('O', 'O1', 'B1', 'B2', 'B3'), 22),
	"Granulocyte_progenitors": (('O', 'O1', 'G1', 'G2'), 20),
	"Granulocytes": (('O', 'O1', 'G1', 'G2', 'G3', 'G4'), 22),
	"Erythroid_progenitors_Erythroblasts": (('O', 'O1', 'E1', 'E2'), 21),
	"Erythrocytes": (('O', 'O1', 'E1', 'E2', 'E3'), 22),
	"Monocyte_progenitors": (('O', 'O1', 'MD1', 'M1'), 20),
	"Monocytes": (('O', 'O1', 'MD1', 'M1', 'M2'), 21),
	"Macrophages": (('O', 'O1', 'Mp1'), 22),
	"Megakaryocyte_progenitors": (('O', 'O1', 'Mk1'), 20),
}


ALL_LABELS["Mouse_Data_Regev_SmartSeq.log2.TPM.Capitalized.rds"] = {
	"TA": (('I1',), 14),
	"Stem": (('I1',), 14),
	"Enterocyte_Progenitor_Early": (('I1', 'I2'), 21),
	"Enterocyte_Progenitor_Late": (('I1', 'I2'), 21),
	"Enterocyte": (('I1', 'I2', 'I3'), 22),
	"Endocrine": (('I1', 'Endocrine'), 22),
	"Tuft": (('I1', 'Tuft'), 22),
	"Goblet": (('I1', 'Goblet'), 22),
	"Paneth": (('I1', 'Paneth'), 22),
}

ALL_LABELS["Intestine_Dropseq.rds"] = {
    'Endocrine': (('I1', 'Endocrine'), 22),
    'Enterocyte_Immature_Distal': (('I1', 'I2', 'I3'), 22),
    'Enterocyte_Immature_Proximal': (('I1', 'I2', 'I3'), 22),
    'Enterocyte_Mature_Distal': (('I1', 'I2', 'I3'), 22),
    'Enterocyte_Mature_Proximal': (('I1', 'I2', 'I3'), 22),
    'Enterocyte_Progenitor': (('I1', 'I2'), 21),
    'Enterocyte_Progenitor_Early': (('I1', 'I2'), 21),
    'Enterocyte_Progenitor_Late': (('I1', 'I2'), 21),
    'Goblet': (('I1', 'Goblet'), 22),
    'Paneth': (('I1', 'Paneth'), 22),
    'Stem': (('I1',), 14),
    'TA_Early': (('I1',), 14),
    'TA_G1': (('I1',), 14),
    'TA_G2': (('I1',), 14),
    'Tuft': (('I1', 'Tuft'), 22),
}

ALL_LABELS["Human_Data_Sample_Blood_AllMerged.rds"] = {
    'Bcells': (('HumanBlood', 'HumanBlood_Bcells'), 22),
    'CD14mono': (('HumanBlood', 'HumanBlood_Bcells'), 22),
    'CD34': (('HumanBlood',), 15),
    'CD4Thelper': (('HumanBlood', 'HumanBlood_CD4Thelper'), 22),
    'CD8T': (('HumanBlood', 'HumanBlood_CD8T'), 22),
    'MemoryCD4T': (('HumanBlood', 'HumanBlood_MemoryCD4T'), 22),
    'NK': (('HumanBlood', 'HumanBlood_NK'), 22),
    'NaiveCD4T': (('HumanBlood', 'HumanBlood_NaiveCD4T'), 22),
    'NaiveCD8T': (('HumanBlood', 'HumanBlood_NaiveCD8T'), 22),
    'Treg': (('HumanBlood', 'HumanBlood_Treg'), 22),
}


COMBINED_LABELS = {}
for k in ALL_LABELS:
    for k2 in ALL_LABELS[k]:
        if k2 in COMBINED_LABELS:
            assert COMBINED_LABELS[k2] == ALL_LABELS[k][k2]
        COMBINED_LABELS[k2] = ALL_LABELS[k][k2]

ABBR_TO_ORDER = {}
for k in COMBINED_LABELS:
    pair = COMBINED_LABELS[k]
    ABBR_TO_ORDER[pair[0]] = pair[1]