import numpy as np
import lifelikefunctions as ll
from migfunctions import *
import os.path
from os import path

# :: Generates a random bitmap
def golric(gridsize, border):

    # : Determine the size of the inner agent
    inner_size = 0
    for i in range(border):
        inner_size = inner_size + gridsize - (2 * (i + 1) - 1)
    inner_size = int(np.sqrt(gridsize**2 - 4*inner_size))

    # : Create randomized inner agent
    inner_agent = np.random.choice(2, inner_size*inner_size, p=[0.5, .5]).reshape(inner_size, inner_size)

    # : Place inner agent into larger field of zeros
    bmi = np.zeros((gridsize, gridsize), dtype=np.int)
    bmi[border:border + inner_agent.shape[0], border:border + inner_agent.shape[1]] = inner_agent

    return bmi

def meanify(tdmlist, gridsize):
    m = []
    for t in range(len(tdmlist)):
        l = tdmlist[tdmlist[:, 0] == t]
        if len(l) > 0:
            l = np.mean(l, axis=0)
            m.append(l)
    m = np.vstack(m)
    return m

# :: Run
def main():

    # psuedolife(B357/S238), drighlife(B367/S23), eppstein(B368/S12578)
    # b356s23, b3568s23, b356s238, b3568s238
    # Part2 of Best Rule: b356s23
    filelist = ['b35s236.csv']
    mfilelist = ['b35s236-means.csv']
    Blist = [{3,5}]
    Slist = [{2, 3, 6}]
    # B0123478/S01234678

    for rule in range(len(Blist)):
        # :: Initializations
        gSamples = 50
        gridsize = 50
        lifesteps = 20
        border = 0
        data = []
        B = Blist[rule]
        S = Slist[rule]

        print(f'BEGIN --- {filelist[rule]}')
        # :: Run Experiment Many Times
        for i in range(gSamples):

            print(i)

            # :: initial and final bitmaps
            bm = golric(gridsize, border)
            neighbors = ll.t_neighbor_sum(bm)

            for t in range(lifesteps):
                bm, neighbors = ll.update(bm, neighbors, B, S)

                # :: measurements
                d = bm.sum()
                
                # :: create random bitmap
                bmr = np.concatenate([np.repeat(1, d), np.repeat(0, gridsize**2 - d)])
                np.random.shuffle(bmr)
                bmr = bmr.reshape((gridsize, gridsize))

                # :: calculate mig difference
                bm_mig_diff = np.mean([Gu(bm), Gd(bm), Gl(bm), Gr(bm)])
                bmr_mig_diff = np.mean([Gu(bmr), Gd(bmr), Gl(bmr), Gr(bmr)])
                mig_diff = bmr_mig_diff - bm_mig_diff
                
                data.append([t, d, mig_diff])
        
        data = np.vstack(data)
        data = data / [1, gridsize**2, 1]
        
        np.savetxt(filelist[rule], data, delimiter = ',', fmt='%f')
        # np.savetxt('size-testing-mig-difference.csv', data, delimiter = ',', fmt='%f')
        # :: ----------------------------------------------------------------------

        means = meanify(data, gridsize)
        np.savetxt(mfilelist[rule], means, delimiter=',', fmt='%f')
        # np.savetxt('size-testing-means.csv', data, delimiter = ',', fmt='%f')
        # :: ----------------------------------------------------------------------
        print(f'COMPLETED --- {filelist[rule]}')

if __name__ == "__main__":
    main()
