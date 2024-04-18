import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy import linalg

class calculate:
    """
    calculate类实现了基于子集选择的线性回归模型。基于给定的训练数据集和测试数据集，
    对所有可能的变量组合进行线性回归，并选择出测试误差最小的模型。

    参数:
    x : ndarray
        自变量的数据矩阵。
    y : ndarray
        因变量的向量。
    index : ndarray
        一个布尔数组，用于指定哪些数据点用于训练。
    names : list
        自变量的名称列表。

    属性:
    xtrain : ndarray
        训练数据集中的自变量。
    xtest : ndarray
        测试数据集中的自变量。
    ytrain : ndarray
        训练数据集中的因变量。
    ytest : ndarray
        测试数据集中的因变量。
    """
    def __init__(self, x, y, index, names):
        """
        初始化OSR对象。
        """
        self.x = x
        self.y = y
        self.index = index
        self.names = names
        self.xtrain, self.xtest = x[self.index], x[~self.index]
        self.ytrain, self.ytest = y[self.index], y[~self.index]
        
        self.xtx = np.dot(self.xtrain.T, self.xtrain)
        self.xty = np.dot(self.xtrain.T, self.ytrain)
        self.xtrain_mean = self.xtrain.mean(axis=0)
        self.xtrain_scale = self.xtrain - self.xtrain_mean
        self.ytrain_mean = self.ytrain.mean()
        self.ytrain_scale = self.ytrain - self.ytrain_mean
        
    def solve_pos(self, xtx, xty):
        """
        使用Cholesky分解解决正定线性方程组。
        
        参数:
        xtx : ndarray
            自变量转置乘以自变量的矩阵。
        xty : ndarray
            自变量转置乘以因变量的向量。

        返回:
        解向量beta。
        """
        mat = linalg.cholesky(xtx)
        return linalg.lapack.dpotrs(mat, xty)[0]
    
    def predict_err(self, x, y, b):
        """
        计算预测误差的平方和。
        
        参数:
        x : ndarray
            测试集的自变量数据。
        y : ndarray
            测试集的因变量数据。
        b : tuple
            线性模型的截距和系数，形式为(b0, b1)。
        
        返回:
        预测误差的平方和。
        """
        b0, b1 = b
        err = y - b0 - np.dot(x, b1)
        return np.inner(err, err)
        
    def osr(self):
        """
        对所有可能的自变量组合执行线性回归，找出测试误差最小的模型。

        返回:
        tuple：包含最优变量组合的名称、回归系数、训练误差、测试误差和模型中变量的数量。
        """
        n, p = self.xtrain.shape
        inds_ = [combinations(range(p), i) for i in range(1, p + 1)]
        inds_1 = [[list(ind_) for ind_ in ind] for ind in inds_]
        p_1 = np.array([len(ind) for ind in inds_1])
        inds = [list(ind_) for ind in inds_1 for ind_ in ind]

        xtrain_mean = self.xtrain.mean(axis=0)
        xtrain_scale = self.xtrain - xtrain_mean
        ytrain_mean = self.ytrain.mean()
        ytrain_scale = self.ytrain - ytrain_mean
        # xtx = np.dot(xtrain_scale.T, xtrain_scale)
        # xty = np.dot(xtrain_scale.T, ytrain_scale)
        xtx = np.dot(self.xtrain.T, self.xtrain)
        xtx_center = xtx - n * np.outer(xtrain_mean, xtrain_mean)
        xty = np.dot(self.xtrain.T, self.ytrain)
        xty_center = xty- n * ytrain_mean * xtrain_mean
        re_b_1 = [self.solve_pos(xtx_center[np.ix_(ind, ind)], xty_center[ind])
                for ind in inds]
        re_b_0 = [ytrain_mean - np.dot(xtrain_mean[ind], b_1_)
                for ind, b_1_ in zip(inds, re_b_1)]
        re_b = [(b_0_, b_1_) for b_0_, b_1_ in zip(re_b_0, re_b_1)]

        # re_b = [ols(xtrain[:, ind], ytrain) for ind in inds]
        err_test = np.array([self.predict_err(self.xtest[:, ind], self.ytest, b_) 
                            for ind, b_ in zip(inds,re_b)])
        rss = np.array([self.predict_err(self.xtrain[:, ind], self.ytrain, b_) 
                        for ind, b_ in zip(inds,re_b)])
        d = np.array(list([len(ind) for ind in inds]))
        # print(d)
        # print(names[list(inds[np.argmin(err_test)])])
        # re_b[np.argmin(err_test)]
        # xty
        return self.names[inds[np.argmin(err_test)]], re_b[np.argmin(err_test)], rss[np.argmin(err_test)], err_test[np.argmin(err_test)], d[np.argmin(err_test)]
    
    def cv_err_i_fun(self, index, inds):
        # index = indexs[0]
        n, p = self.xtrain.shape
        n_i = np.size(index)
        n_i_ = n - n_i
        x_i = self.xtrain[index, :]
        xt_i = x_i.T
        y_i = self.ytrain[index]
        x_mean_i = np.mean(x_i, axis=0)
        y_mean_i = np.mean(y_i)
        xtx_i = np.dot(xt_i, x_i) - n_i * np.outer(x_mean_i, x_mean_i)
        xty_i = np.dot(xt_i, y_i) - n_i * y_mean_i * x_mean_i
        
        xtrain_mean = self.xtrain.mean(axis=0)
        xtrain_scale = self.xtrain - xtrain_mean
        ytrain_mean = self.ytrain.mean()
        ytrain_scale = self.ytrain - ytrain_mean

        

        x_mean_i_ = (n*xtrain_mean - n_i*x_mean_i) / n_i_
        y_mean_i_ = (n*ytrain_mean - n_i*y_mean_i) / n_i_
        x_mean_i__ = x_mean_i - xtrain_mean
        y_mean_i__ = y_mean_i - ytrain_mean
        xtx_i_ = self.xtx - xtx_i - (n_i*n/n_i_) * np.outer(x_mean_i__, x_mean_i__)
        xty_i_ = self.xty - xty_i - (n_i*n/n_i_) * y_mean_i__ * x_mean_i__
        re_b_1_i = [self.solve_pos(xtx_i_[np.ix_(ind, ind)], xty_i_[ind]) 
                    for ind in inds]
        re_b_0_i = [y_mean_i_ - np.inner(x_mean_i_[ind], b_1_i) 
                    for ind, b_1_i in zip(inds, re_b_1_i)]
        cv_err_i = [self.predict_err(x_i[:, ind], y_i, (b_0_i, b_1_i))
                    for ind, b_0_i, b_1_i in zip(inds, re_b_0_i, re_b_1_i)]
        return cv_err_i
    
    def cross_validation(self, K):
        n, p = self.xtrain.shape
        
        indexs = np.array_split(np.random.permutation(np.arange(n)), K)
        
        inds_ = [combinations(range(p), i) for i in range(1, p + 1)]
        inds_1 = [[list(ind_) for ind_ in ind] for ind in inds_]
        p_1 = np.array([len(ind) for ind in inds_1])
        inds = [list(ind_) for ind in inds_1 for ind_ in ind]
        
        cv_err_mat = np.array([self.cv_err_i_fun(index, inds) for index in indexs])
        # print(cv_err_mat.shape)
        cv_err = cv_err_mat.sum(axis=0) / n
        print(self.names[inds[np.argmin(cv_err)]])
        return np.min(cv_err), self.names[inds[np.argmin(cv_err)]] 
    
    
if __name__ == '__main__':
    x = np.loadtxt("data/x.txt", delimiter=",")
    y = np.loadtxt("data/y.txt", delimiter=",")
    index = np.loadtxt("data/index.txt", delimiter=",", dtype=bool)
    names = np.loadtxt("data/names.txt", delimiter=",", dtype=str)
    analysis = calculate(x, y, index, names)
    res = analysis.osr()
    print(res)   


