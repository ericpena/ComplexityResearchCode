# :: migfunctions.py

# :: preliminaries
import numpy as np

def Gu(bitmap):
    """
	Initialize a population of Agents

	Args:
		bitmap (np.array): 2D Grid
	Returns:
		(float): mean information gain upward
	"""
    pbs = bitmap[1:,:].sum() / ((bitmap.shape[0] - 1) * bitmap.shape[1])
    pws = ((bitmap.shape[0] - 1) * bitmap.shape[1] - bitmap[1:,:].sum()) / ((bitmap.shape[0] - 1) * bitmap.shape[1])

    # :: idx for black and white cells
    rb, cb = np.where(bitmap == 1)
    rw, cw = np.where(bitmap == 0)

    # :: take rows below top row
    zb = np.count_nonzero(rb == 0)
    rb = rb[zb:]
    cb = cb[zb:]
    zw = np.count_nonzero(rw == 0)
    rw = rw[zw:]
    cw = cw[zw:]

    # :: create list of coordinates
    lb = list(zip(rb, cb))
    lw = list(zip(rw, cw))

    # :: cells above black and white cells
    oneaboveb = np.array([bitmap[r - 1, c] for r, c in lb])
    oneabovew = np.array([bitmap[r - 1, c] for r, c in lw])

    # :: numerator
    numbb = np.count_nonzero(oneaboveb == 1)
    numww = np.count_nonzero(oneabovew == 0)
    numbw = np.count_nonzero(oneaboveb == 0)
    numwb = np.count_nonzero(oneabovew == 1)

    # :: denominator
    denb = np.sum(bitmap[1:,:])
    denw = (bitmap.shape[0] - 1) * bitmap.shape[1] - np.sum(bitmap[1:,:])

    pbgb = numbb / denb
    pwgw = numww / denw
    pbgw = numbw / denb
    pwgb = numwb / denw

    pbb = pbs * pbgb
    pww = pws * pwgw
    pbw = pbs * pbgw
    pwb = pws * pwgb

    Gbb = 0 if pbgb == 0 else -pbb * np.log2(pbgb)
    Gww = 0 if pwgw == 0 else -pww * np.log2(pwgw)
    Gbw = 0 if pbgw == 0 else -pbw * np.log2(pbgw)
    Gwb = 0 if pwgb == 0 else -pwb * np.log2(pwgb)

    return (Gbb + Gww + Gbw + Gwb)

def Gd(bitmap):
    """
	Initialize a population of Agents

	Args:
		bitmap (np.array): 2D Grid
	Returns:
		(float): mean information gain downward
	"""
    pbs = bitmap[:(bitmap.shape[0] - 1),:].sum() / ((bitmap.shape[0] - 1) * bitmap.shape[1])
    pws = ((bitmap.shape[0] - 1) * bitmap.shape[1] - bitmap[:(bitmap.shape[0] - 1),:].sum()) / ((bitmap.shape[0] - 1) * bitmap.shape[1])

    # :: idx for black and white cells
    rb, cb = np.where(bitmap == 1)
    rw, cw = np.where(bitmap == 0)

    # :: take rows above bottom row
    zb = len(rb) - np.count_nonzero(rb == (bitmap.shape[0] - 1))
    rb = rb[:zb]
    cb = cb[:zb]
    zw = len(rw) - np.count_nonzero(rw == (bitmap.shape[0] - 1))
    rw = rw[:zw]
    cw = cw[:zw]

    lb = list(zip(rb, cb))
    lw = list(zip(rw, cw))

    onebelowb = np.array([bitmap[r + 1, c] for r, c in lb])
    onebeloww = np.array([bitmap[r + 1, c] for r, c in lw])

    # :: numerator
    numbb = np.count_nonzero(onebelowb == 1)
    numww = np.count_nonzero(onebeloww == 0)
    numbw = np.count_nonzero(onebelowb == 0)
    numwb = np.count_nonzero(onebeloww == 1)

    # :: denominator
    denb = np.sum(bitmap[:(bitmap.shape[0] - 1),:])
    denw = (bitmap.shape[0] - 1) * bitmap.shape[1] - np.sum(bitmap[:(bitmap.shape[0] - 1),:])

    pbgb = numbb / denb
    pwgw = numww / denw
    pbgw = numbw / denb
    pwgb = numwb / denw

    pbb = pbs * pbgb
    pww = pws * pwgw
    pbw = pbs * pbgw
    pwb = pws * pwgb

    Gbb = 0 if pbgb == 0 else -pbb * np.log2(pbgb)
    Gww = 0 if pwgw == 0 else -pww * np.log2(pwgw)
    Gbw = 0 if pbgw == 0 else -pbw * np.log2(pbgw)
    Gwb = 0 if pwgb == 0 else -pwb * np.log2(pwgb)

    return (Gbb + Gww + Gbw + Gwb)

def Gl(bitmap):
    """
	Initialize a population of Agents

	Args:
		bitmap (np.array): 2D Grid
	Returns:
		(float): mean information gain lefd-ward
	"""
    bitmap = np.transpose(bitmap)

    pbs = bitmap[1:,:].sum() / ((bitmap.shape[0] - 1) * bitmap.shape[1])
    pws = ((bitmap.shape[0] - 1) * bitmap.shape[1] - bitmap[1:,:].sum()) / ((bitmap.shape[0] - 1) * bitmap.shape[1])

    # :: idx for black and white cells
    rb, cb = np.where(bitmap == 1)
    rw, cw = np.where(bitmap == 0)

    # :: take rows below top row
    zb = np.count_nonzero(rb == 0)
    rb = rb[zb:]
    cb = cb[zb:]
    zw = np.count_nonzero(rw == 0)
    rw = rw[zw:]
    cw = cw[zw:]

    # :: create list of coordinates
    lb = list(zip(rb, cb))
    lw = list(zip(rw, cw))

    # :: cells above black and white cells
    oneleftb = np.array([bitmap[r - 1, c] for r, c in lb])
    oneleftw = np.array([bitmap[r - 1, c] for r, c in lw])

    # :: numerator
    numbb = np.count_nonzero(oneleftb == 1)
    numww = np.count_nonzero(oneleftw == 0)
    numbw = np.count_nonzero(oneleftb == 0)
    numwb = np.count_nonzero(oneleftw == 1)

    # :: denominator
    denb = np.sum(bitmap[1:,:])
    denw = (bitmap.shape[0] - 1) * bitmap.shape[1] - np.sum(bitmap[1:,:])

    pbgb = numbb / denb
    pwgw = numww / denw
    pbgw = numbw / denb
    pwgb = numwb / denw

    pbb = pbs * pbgb
    pww = pws * pwgw
    pbw = pbs * pbgw
    pwb = pws * pwgb

    Gbb = 0 if pbgb == 0 else -pbb * np.log2(pbgb)
    Gww = 0 if pwgw == 0 else -pww * np.log2(pwgw)
    Gbw = 0 if pbgw == 0 else -pbw * np.log2(pbgw)
    Gwb = 0 if pwgb == 0 else -pwb * np.log2(pwgb)

    return (Gbb + Gww + Gbw + Gwb)

def Gr(bitmap):
    """
	Initialize a population of Agents

	Args:
		bitmap (np.array): 2D Grid
	Returns:
		(float): mean information gain right-ward
	"""
    bitmap = np.transpose(bitmap)

    pbs = bitmap[:(bitmap.shape[0] - 1),:].sum() / ((bitmap.shape[0] - 1) * bitmap.shape[1])
    pws = ((bitmap.shape[0] - 1) * bitmap.shape[1] - bitmap[:(bitmap.shape[0] - 1),:].sum()) / ((bitmap.shape[0] - 1) * bitmap.shape[1])

    # :: idx for black and white cells
    rb, cb = np.where(bitmap == 1)
    rw, cw = np.where(bitmap == 0)

    # :: take rows above bottom row
    zb = len(rb) - np.count_nonzero(rb == (bitmap.shape[0] - 1))
    rb = rb[:zb]
    cb = cb[:zb]
    zw = len(rw) - np.count_nonzero(rw == (bitmap.shape[0] - 1))
    rw = rw[:zw]
    cw = cw[:zw]

    lb = list(zip(rb, cb))
    lw = list(zip(rw, cw))

    onebelowb = np.array([bitmap[r + 1, c] for r, c in lb])
    onebeloww = np.array([bitmap[r + 1, c] for r, c in lw])

    # :: numerator
    numbb = np.count_nonzero(onebelowb == 1)
    numww = np.count_nonzero(onebeloww == 0)
    numbw = np.count_nonzero(onebelowb == 0)
    numwb = np.count_nonzero(onebeloww == 1)

    # :: denominator
    denb = np.sum(bitmap[:(bitmap.shape[0] - 1),:])
    denw = (bitmap.shape[0] - 1) * bitmap.shape[1] - np.sum(bitmap[:(bitmap.shape[0] - 1),:])

    pbgb = numbb / denb
    pwgw = numww / denw
    pbgw = numbw / denb
    pwgb = numwb / denw

    pbb = pbs * pbgb
    pww = pws * pwgw
    pbw = pbs * pbgw
    pwb = pws * pwgb

    Gbb = 0 if pbgb == 0 else -pbb * np.log2(pbgb)
    Gww = 0 if pwgw == 0 else -pww * np.log2(pwgw)
    Gbw = 0 if pbgw == 0 else -pbw * np.log2(pbgw)
    Gwb = 0 if pwgb == 0 else -pwb * np.log2(pwgb)

    return (Gbb + Gww + Gbw + Gwb)