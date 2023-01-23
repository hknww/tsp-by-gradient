# %%
import tsp_gradient_outils as tsp
from tqdm import tqdm

# %%
liste = ["004", "005", "006", "008", "010", "012", "016", "024", "032", "040", "060", "080", "100", "120", "180", "220", "300", "420", "560", "800"]

# %%
for a in liste[0:7]:
    for i in tqdm(range(1000)):
        P_1 = tsp.TSPProblem(int(a), int(a) * 2)
        file_name = "data/size_" + a + "/instance_" + a + "_" + "{:03d}".format(i) + ".csv"
        try:
            P_1.load(file_name)
        except:
            P_1.lp_solve()
            P_1.save(file_name)


# %%
for a in liste[7:10]:
    for i in tqdm(range(50)):
        P_1 = tsp.TSPProblem(int(a), int(a) * 2)
        file_name = "data/size_" + a + "/instance_" + a + "_" + "{:03d}".format(i) + ".csv"
        try:
            P_1.load(file_name)
        except:
            P_1.lp_solve()
            P_1.save(file_name)



