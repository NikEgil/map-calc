from scipy.interpolate import RBFInterpolator
import numpy as np
import pandas as pd
import copy, warnings, scipy
from sklearn.linear_model import RidgeCV

''' Файл является кусочком из библиотеки pyfitit '''
def scoreFast(y, predictY):
    if len(y.shape) >= 2 and np.sum(y.shape != 1) == 1:
        y = y.flatten()
    if len(predictY.shape) >= 2 and np.sum(predictY.shape != 1) == 1:
        predictY = predictY.flatten()
    assert np.all(y.shape == predictY.shape), f"{y.shape} != {predictY.shape}"
    if len(y.shape) == 1:
        u = np.mean((y - predictY) ** 2)
        v = np.mean((y - np.mean(y)) ** 2)
    else:
        u = np.mean(np.linalg.norm(y - predictY, axis=1, ord=2) ** 2)
        v = np.mean(
            np.linalg.norm(
                y - np.mean(y, axis=0).reshape([1, y.shape[1]]), axis=1, ord=2
            )
            ** 2
        )
    if v == 0:
        return 0
    return 1 - u / v


def score(x, y, predictor):
    predictY = predictor(x)
    return scoreFast(y, predictY)


def getMinDist(x1, x2):
    NNdists, mean = getNNdistsStable(x1, x2)
    # take min element with max index (we want to throw last points first)
    min_dist_ind = np.max(np.where(NNdists == np.min(NNdists))[0])
    min_dist = NNdists[min_dist_ind]
    return min_dist_ind, min_dist, mean


def unique_mulitdim(p, rel_err=1e-6):
    """
    Remove duplicate points from array p

    :param p: points (each row is one point)
    :param rel_err: max difference between points to be equal i.e. dist < np.quantile(all_dists, 0.1) * rel_err
    :returns: new_p, good_ind
    """
    p = copy.deepcopy(p)
    assert (
        len(p.shape) == 2
    ), "p must be 2-dim (each row is one point). For 1 dim we can't determine whether it is row or column"
    good_ind = np.arange(len(p))
    min_dist_ind, min_dist, med_dist = getMinDist(p, p)
    while min_dist <= med_dist * rel_err:
        good_ind = np.delete(good_ind, min_dist_ind)
        min_dist_ind, min_dist, _ = getMinDist(p[good_ind, :], p[good_ind, :])
    return p[good_ind, :], good_ind


def getNNdistsStable(x1, x2):
    dists = scipy.spatial.distance.cdist(x1, x2)
    M = np.max(dists)
    mean = np.mean(dists)
    np.fill_diagonal(dists, M)
    NNdists = np.min(dists, axis=1)
    return NNdists, mean


def getWeightsForNonUniformSample(x):
    """
    Calculates weights for each object x[i] = NN_dist^dim. These weights make uniform the error of ML models fitted on non-uniform samples
    """
    assert len(x.shape) == 2
    if len(x) <= 1:
        return np.ones(len(x))
    NNdists, _ = getNNdistsStable(x, x)
    w = NNdists ** x.shape[1]
    w /= np.sum(w)
    w[w < 1e-6] = 1e-6
    w /= np.sum(w)
    return w


class RBF:
    def __init__(
        self,
        function="linear",
        baseRegression="quadric",
        scaleX=True,
        removeDublicates=False,
    ):
        """
        RBF predictor
        :param function: string. Possible values: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        :param baseRegression: string, base estimator. Possible values: quadric, linear, None
        :param scaleX: bool. Scale X by gradients of y
        """
        # function: multiquadric, inverse, gaussian, linear, cubic, quintic, thin_plate
        # baseRegression: linear quadric
        self.function = function
        self.baseRegression = baseRegression
        self.trained = False
        self.scaleX = scaleX
        self.train_x = None
        self.train_y = None
        self.base = None
        self.scaleGrad = None
        self.minX = None
        self.maxX = None
        self.interp = None
        self.removeDublicates = removeDublicates

    def get_params(self, deep=True):
        return {
            "function": self.function,
            "baseRegression": self.baseRegression,
            "scaleX": self.scaleX,
        }

    def set_params(self, **params):
        self.function = copy.deepcopy(params["function"])
        self.baseRegression = copy.deepcopy(params["baseRegression"])
        self.scaleX = copy.deepcopy(params["scaleX"])
        return self

    def fit(self, x, y):
        x = copy.deepcopy(x)
        y = copy.deepcopy(y)
        self.train_x = (
            x.values if (type(x) is pd.DataFrame) or (type(x) is pd.Series) else x
        )
        self.train_y = (
            y.values if (type(y) is pd.DataFrame) or (type(y) is pd.Series) else y
        )
        if len(self.train_y.shape) == 1:
            self.train_y = self.train_y.reshape(-1, 1)
        if self.baseRegression == "quadric":
            self.base = makeQuadric(RidgeCV())
        elif self.baseRegression is None:
            self.base = None
        else:
            assert self.baseRegression == "linear"
            self.base = RidgeCV()
        if self.scaleX:
            n = self.train_x.shape[1]
            self.minX = np.min(self.train_x, axis=0)
            self.maxX = np.max(self.train_x, axis=0)
            self.train_x = norm(self.train_x, self.minX, self.maxX)
            quadric = makeQuadric(RidgeCV())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                quadric.fit(self.train_x, self.train_y)
            center_x = np.zeros(n)
            center_y = quadric.predict(center_x.reshape(1, -1))
            grad = np.zeros(n)
            for i in range(n):
                h = 1
                x2 = np.copy(center_x)
                x2[i] = center_x[i] + h
                y2 = quadric.predict(x2.reshape(1, -1))
                x1 = np.copy(center_x)
                x1[i] = center_x[i] - h
                y1 = quadric.predict(x1.reshape(1, -1))
                grad[i] = np.max(
                    [
                        np.linalg.norm(y2 - center_y, ord=np.inf) / h,
                        np.linalg.norm(center_y - y1, ord=np.inf) / h,
                    ]
                )
            if np.max(grad) == 0:
                if self.train_x.shape[0] > 2:
                    warnings.warn(
                        f"Constant function. Gradient = 0. x.shape={self.train_x.shape}"
                    )
                self.scaleGrad = np.ones((1, n))
            else:
                grad = grad / np.max(grad)
                eps = 0.01
                if len(grad[grad <= eps]) > 0:
                    grad[grad <= eps] = np.min(grad[grad > eps]) * 0.01
                self.scaleGrad = grad.reshape(1, -1)
                self.train_x = self.train_x * self.scaleGrad
        if self.removeDublicates:
            # RBF crashes when dataset includes close or equal points
            self.train_x, uniq_ind = unique_mulitdim(self.train_x)
            self.train_y = self.train_y[uniq_ind, :]
        w = getWeightsForNonUniformSample(self.train_x)
        if self.baseRegression is not None:
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                self.base.fit(self.train_x, self.train_y, sample_weight=w)
            bp = self.base.predict(self.train_x)
            self.train_y = self.train_y - bp.reshape(self.train_y.shape)
        NdimsY = self.train_y.shape[1]
        assert NdimsY > 0
        self.interp = RBFInterpolator(
            self.train_x, self.train_y, kernel=self.function, degree=0
        )
        self.trained = True

    def predict(self, x):
        assert self.trained
        if type(x) is pd.DataFrame:
            x = x.values
        assert len(x.shape) == 2, f"x = " + str(x)
        assert (
            x.shape[1] == self.train_x.shape[1]
        ), f"{x.shape[1]} != {self.train_x.shape[1]}"
        if self.scaleX:
            x = norm(x, self.minX, self.maxX)
            x = x * self.scaleGrad

        res = self.interp(x)

        assert res.shape[0] == x.shape[0], f"{res.shape[0]} != {x.shape[0]}"
        if self.baseRegression is not None:
            res = res + self.base.predict(x).reshape(res.shape)
        return res

    def score(self, x, y):
        return score(x, y, self.predict)


class RBFWrapper(RBF):
    def predict(self, x):
        result = RBF.predict(self, x).flatten()
        return result


def transformFeatures2Quadric(x, addConst=True):
    isDataframe = type(x) is pd.DataFrame
    if isDataframe:
        col_names = np.array(x.columns)
        x = x.values
    n = x.shape[1]
    new_n = n + n * (n + 1) // 2
    if addConst:
        new_n += 1
    newX = np.zeros([x.shape[0], new_n])
    newX[:, :n] = x
    if isDataframe:
        new_col_names = np.array([""] * newX.shape[1], dtype=object)
        new_col_names[:n] = col_names
    k = n
    for i1 in range(n):
        for i2 in range(i1, n):
            newX[:, k] = x[:, i1] * x[:, i2]
            if isDataframe:
                if i1 != i2:
                    new_col_names[k] = col_names[i1] + "*" + col_names[i2]
                else:
                    new_col_names[k] = col_names[i1] + "^2"
            k += 1
    if addConst:
        newX[:, k] = 1
        if isDataframe:
            new_col_names[k] = "const"
        k += 1
        assert k == n + n * (n + 1) // 2 + 1
    else:
        assert k == n + n * (n + 1) // 2
    if isDataframe:
        newX = pd.DataFrame(newX, columns=new_col_names)
    return newX


class makeQuadric:
    def __init__(self, learner):
        self.learner = learner

    def get_params(self, deep=True):
        return {"learner": self.learner}

    def set_params(self, **params):
        self.learner = copy.deepcopy(params["learner"])
        return self

    def fit(self, x, y, **args):
        x2 = transformFeatures2Quadric(x)
        self.learner.fit(x2, y, **args)

    def predict(self, x):
        return self.learner.predict(transformFeatures2Quadric(x))

    def score(self, x, y):
        return score(x, y, self.predict)


def norm(x, minX, maxX):
    """
    Do not norm columns in x for which minX == maxX
    :param x:
    :param minX:
    :param maxX:
    :return:
    """
    dx = maxX - minX
    ind = dx != 0
    res = copy.deepcopy(x)
    if type(x) is pd.DataFrame:
        res.loc[:, ind] = 2 * (x.loc[:, ind] - minX[ind]) / dx[ind] - 1
        res.loc[:, ~ind] = 0
    else:
        if minX.size == 1:
            if dx != 0:
                res = 2 * (x - minX) / dx - 1
            else:
                res[:] = 0
        else:
            res[:, ind] = 2 * (x[:, ind] - minX[ind]) / dx[ind] - 1
            res[:, ~ind] = 0
    return res
