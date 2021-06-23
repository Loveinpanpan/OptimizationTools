"""
求解经典的一般线性规划问题
按照标准型进行求解（目标函数最大化、线性等式约束，所有变量非负）
"""
import numpy as np
from itertools import combinations
from sympy.Matrix import rref

class simplex():
    def __init__(self, c, a, b, x0, bx_index, z0):
        """
        c: 目标函数系数向量(目标函数Max)
        a: 变量约束系数矩阵，“=”左边
        b: 约束条件中的非负常量， “=”右边
        x0: 初始基可行解
        bx_index：基变量序号
        z0：初始解条件下的目标函数值
        """
        self.C = c #初始变量的系数
        self.A = a
        self.B = b
        self.X0 = x0
        self.BX_index = bx_index
        self.Z0 = z0
        self.stop_gap = 1.0e-6
    
    # 计算非基变量检验数
    def _non_x_test(self, bx_index, nx_index):
        #bx_index:基变量序号
        #nx_index：非基变量序号
        
        opt_tag = False
        
        c_n = np.array([self.C[x] for x in nx_index])
        c_b = np.array([self.C[x] for x in bx_index])
        matrix_A = np.array(self.A)
        
        matrix_b = np.zeros((len(self.B), len(bx_index))#基向量矩阵
        
        matrix_n = np.zeros((len(self.B), len(nx_index)) #非基向量矩阵
        
        for i, x_index in enumerate(bx_index):
            matrix_b[:,i] = matrix_A[:,x_index]
        
        for i, x_index in enumerate(nx_index):
            matrix_n[:,i] = matrix_A[:,x_index]
        # 计算检验系数
        nb_x_test = c_n - np.dot(np.dot(c_b,np.linalg.inv(matrix_b)),matrix_n)
        
        nb_max_test = max(nb_x_test)
        
        if nb_max_test <= 0:
            opt_tag = True
            return opt_tag, None, None
        
        nb_x_in_index = None#可进入基变量的待选非基变量，存储入基变量的序列号
        for j, x_test in enumerate(nb_x_test):
            if nb_x_in_index != None:
                break
            if x_test == nb_max_test:
                nb_x_in_index = nx_index[j]
        nb_x_in_a = matrix_n[:,nb_x_test.index(nb_x_in_index)]
        
        # 计算theta
        theta_list = []
        for j, x_index in enumerate(bx_index):
            if nb_x_in_a[j] > 0:
                theta = self.B[j]/nb_x_in_a[j]
                theta_list.append([theta,x_index])
        x_out_index_list = [y[1] for y in sorted(theta_list,key=(lambda x:x[0]),reverse=False)]#按theta值正序排序（从小到大）
        return opt_tag, nb_x_in_index, x_out_index_list
    
    
    # 依据更新后的基变量，重新生成一组可行解
    def _b_x_solve(self, bx_index):
        x0 = [0]*len(self.C)
        nb_x_index = [x for x in range(len(self.C)) if x not in bx_index]
        A = np.array(self.A)
        matrix_b = np.zeors((len(self.B), len(bx_index)))
        mtrix_nb = np.zeros((len(self.B), len(nb_x_index)))
        
        
        for i, i_index in enumerate(bx_index):
            matrix_b[:,i] = A[:,i_index]
        
        if (np.linalg.det(matrix_b) == 0):
            return None, None, None
        
        b_x = list(np.linalg.inv(matrix_b)*np.array(self.B))
        
        negtive_count = 0
        
        for i, x_value in enumerate(b_x):
            x0[bx_index[i]] = x_value
            if (x_value < 0):
                negtive_count +=1
        if (negtive_count > 0):
            return None, None, None
        
        z0 = sum(np.array(x0)*np.array(self.C))
        return x0,z0, nb_x_index
    
    # 单纯形法
    def _solve(self):
        
        X=self.X0
        base_x_index = self.BX_index#记录基变量序号
        Z=self.Z0
        Nonbase_x_index = [index for index in range(len(self.C)) if index not in X_index]
        while True:
            #通过非基变量检验数，判断当前可行基是否为最优解
            #当非基变量检验数均<=0时，当前可行基即为最优解
            is_opt, x_in_index, x_out_index_list = self._non_x_test(base_x_index, Nonbase_x_index)
            if is_opt:
                break
            else:
                #更新基变量，并给出一组可行解
                
                #base_x_index[base_x_index.index(x_out_index_list[0])] = x_in_index
                for x_out_index in x_out_index_list:
                    base_x_index_temp = base_x_index.copy()
                    base_x_index_temp[base_x_index_temp.index(x_out_index)] = x_in_index
                    x_temp, z_temp,nb_x_index_temp = self._b_x_solve(base_x_index_temp)
                    if x_temp != None:
                        break
                base_x_index = base_x_index_temp
                
                if (abs(z_temp - X) <= self.stop_gap):
                    X = x_temp
                    Z = z_temp
                    break
                elif (x_temp == None):
                    break
                else:
                    X=x_temp
                    Z = z_temp
                    Nonbase_x_index = nb_x_index_temp
                
        return X, Z, base_x_index


class lineprog():
    def __init__(self, c=None, equal_c=None, equal_b=None, inequal_c=None, inequal_b=None, inequal_above_c=None, inequal_above_b=None, bounds=None):
        """
        c: 目标函数系数向量(目标函数Max)
        equal_c：等式约束的变量系数矩阵，“=”左边
        equal_b：等式约束的常量向量，“=”右边
        inequal_c：不等式约束的变量系数矩阵，“<=”左边
        inequal_b：不等式约束的常量向量，“<=”右边
        inequal_above_c: 不等式约束的变量系数矩阵，“>=”左边
        inequal_above_b: 不等式约束的常量向量，“>=”右边
        bounds：变量范围
        """
        self.c = c #初始变量的系数
        self.equal_c = equal_c
        self.equal_b = equal_b
        self.inequal_c = inequal_c
        self.inequal_b = inequal_b
        self.inequal_above_c = inequal_above_c
        self.inequal_above_b = inequal_above_b
        self.bounds = bounds
        self.stop_gap = 6.0e-6
        #self.X0 = x0
    
    # 标准化
    def standard_trans(self):
        assert self.c != None, 'please input objective function!'
        
        var_num = len(self.c)
        self.varible_num = var_num#原有变量的数量
        
        if self.inequal_above_c == None:
            self.inequal_above_c = []
            self.inequal_above_b = []
        
        if self.inequal_c == None:
            self.inequal_c = []
            self.inequal_b = []
        
        if self.bounds != None:
            assert len(self.bounds) == var_num, 'dims of inputed bounds not equal objective funtion'
            # 检查变量取值范围
            for x_index, x_range in enumerate(self.bounds):
                if x_range[1] != None:
                    vector_l = [0]*var_num
                    vector_l[x_index] = 1
                    self.inequal_c.append(vector_l)
                    self.inequal_b.append(x_range[1])
                if x_range[0] > 0:
                    vector_u = [0]*var_num
                    vector_u[x_index] = 1
                    self.inequal_above_c.append(vector_u)
                    self.inequal_above_b.append(x_range[0])
                    
        self.C = self.c
        var_insert_num = len(self.inequal_b)+len(self.inequal_above_b)
        if var_insert_num!=0:
            self.C = self.C + [0]*var_insert_num
        
        self.A = []
        self.B = []
        
        #针对“>=”和“<=”的约束条件，添加非负松弛变量,转成标准型方程
        if self.equal_c != None:
            for i_index, a_vector in enumerate(self.equal_c):
                self.A.append(a_vector + [0]*var_insert_num)
                self.B.append(self.equal_b[i_index])
        
        if len(self.inequal_c) > 0:
            for inequal_index, inequal_c_l in enumerate(self.inequal_c):
                vector_insert = [0]*var_insert_num
                vector_insert[inequal_index] = 1
                self.A.append(inequal_c_l + vector_insert)
                self.B.append(self.inequal_b[inequal_index])
        back_num = len(self.inequal_c)
        if len(self.inequal_above_c) > 0:
            for inequal_index, inequal_c_u in enumerate(self.inequal_above_c):
                vector_insert = [0]*var_insert_num
                vector_insert[inequal_index+back_num] = -1
                self.A.append(inequal_c_u + vector_insert)
                self.B.append(inequal_above_b[inequal_index])
        

    # 初始可行基
    def initial_x(self):
        M = len(self.B)#初始可行基的方阵大小
        N = len(self.A[0])#变量数量
        
        # 确定矩阵A是否为满秩，即约束条件是否为相互独立
        
        r_A = np.linalg.matrix_rank(self.A)
        
        # 当约束条件非满秩时，即存在约束条件可由其他条件线性相关，
        # 此时采用初等行变换，确定线性相关的约束并剔除，重新对输入的原始模型进行标准化
        if r_A != M:
            return None, None, None
        
        
        #当输入约束条件为标准型时（均满足“<=”），则取松弛变量为初始可行基解
        if ((self.equal_c == None) and (len(self.inequal_above_c) == 0) and (len(inequal_c) > 0)):
            real_var_num = len(self.c)
            total_var_index = [x for x in range(len(C))]
            b_var_index = total_var_index[real_var_num:]
            b_x_matrix = np.zeros((M, M))
            for i, x_index in enumerate(b_var_index):
                b_x_matrix[:,i] = self.A[:,x_index]
            b_x = list(np.dot(np.linalg.inv(b_x_matrix),np.array(self.B)))
            
            x0=[0]*N
            for i, x_index in enumerate(b_var_index):
                x0[x_index] = b_x[i]
            z0 = self.C * x0
            return x0,b_var_index, z0
        
        elif ((self.equal_c == None) or (len(self.inequal_above_c) == 0)):
            #从系数矩阵A中提取M*M的可逆方阵 np.linalg.det()求行列式, np.linalg.inv()
            var_index = [x for x in range(N)]
            x0=[0]*N
            z0=-9999999.0
            tag = 0
            b_var_index = []
            for index_list in combinations(var_index, M):
                a_matrix = np.zeros((M,M))
                for i, col_index in enumerate(index_list):
                    a_matrix[:,i] = self.A[:,col_index]
                    
                if np.linalg.det(a_matrix) == 0:
                    x0=[0]*N
                    continue
                else:
                    a_matrix_inv = np.linalg.inv(a_matrix)
                    b_x = list(np.dot(a_matrix_inv, np.array(self.B)))
                    if len([x for x in b_x if x < 0]) > 0:
                        x0=[0]*N
                        continue
                    else:
                        for m, b_x_index in enumerate(index_list):
                            x0[b_x_index] = b_x[m]
                        b_var_index = index_list
                        z0 = x0*self.C
                        tag = 1
                if tag == 1:
                    break
                    
            if len([x for x in x0 if x == 0]) == N:
                return None,None,None            
            return x0, b_var_index, z0
    
    # 在标准型基础上，增加人工变量，构造第一阶段模型
    # 为了减少人工变量，再构造过程中尽可能多地使用松弛变量
    def _artificial_trans(self):
        A=self.A.copy()
        C_ART = [0]*len(self.C)
        A_ART = self.A.copy()
        B_ART = self.B.copy()
        var_index = [x for x in range(len(self.C))]
        slack_var_index = []
        artificial_num = 0
        if ((self.equal_c == None) and (len(self.inequal_c) > 0)):
            slack_var_index = [x for x in range(0, len(self.equal_c)+len(self.inequal_c))]
        elif ((self.equal_c != None) and (len(self.inequal_c) > 0))
            slack_var_index = [x for x in range(len(self.equal_c), len(self.equal_c)+len(self.inequal_c))]
        
        """
        if len(slack_var_index) > 0:
            for x_index in slack_var_index:
                C_ART[x_index] = 1
        """
        
        if self.equal_c != None:
            artificial_num = len(self.equal_c)
        artificial_num = artificial_num + len(self.inequal_above_c)
        
        if artificial_num > 0:
            A_ART = []
            C_ART = C_ART + [-1]*artificial_num
            row_index = -1
            equal_num = 0
            if self.equal_c != None:
                equal_num = len(elf.equal_c)
                for row_index in range(len(self.equal_c)):
                    art_c_list = [0]*artificial_num
                    art_c_list[row_index] = 1
                    A_ART.append(A[row_index] + art_c_list)
            if len(self.inequal_c) > 0:
                for row_index1 in range(row_index+1, equal_num + len(self.inequal_c)):
                    art_c_list = [0]*artificial_num
                    A_ART.append(A[row_index1] + art_c_list)
            if len(self.inequal_above_c) > 0:
                row_list = [x for x in range(equal_num + len(self.inequal_c), equal_num + len(self.inequal_c) + len(self.inequal_above_c))]
                for index, row_index2 in enumerate(row_list):
                    art_c_list = [0]*artificial_num
                    art_c_list[index + equal_num] = 1
                    A_ART.append(A[row_index] + art_c_list)
            bx_index = slack_var_index + [var_index[-1] + x for x in range(1,artificial_num+1)]
        else:
            bx_index = slack_var_index
            
        bx_matrix = np.zeros((len(bx_index), len(bx_index)))
        for seq, index in enumerate(bx_index):
            bx_matrix[:,seq] = np.array(A_ART)[:,index]
        
        b_x = np.dot(np.linalg.inv(bx_matrix), np.array(self.B))
        x0 = [0]*len(C_ART)
        
        for i, index in enumerate(bx_index):
            x0[index] = b_x[i]
        z0 = sum(np.array(x0) * np.arry(self.C))
        return C_ART,A_ART,B_ART, x0, z0, bx_index, artificial_num
        
    
    # 两阶段法
    def _twoPhase(self):
        # 构造第一阶段模型，并给出初始可行基解
        C_ART,A_ART,B_ART, x0, z0, bx_index, artificial_num = self._artificial_trans()
        if artificial_num:
        
            # 采用单纯形法求解第一阶段模型
            first_solver = simplex(C_ART, A_ART, B_ART, x0,bx_index,z0)
            X0,Z0, base_x_index = first_solver._solve()
            
            # 判断第一阶段的求解结果是否满足进一步求解条件
            artificial_var = [x for x in range(len(C_ART))][-artificial_num:]
            if len([x for x in artificial_var if x in base_x_index]) > 0:
                return None, None
            x0_reset = X0[:-artificial_num]
            z0_reset = x0_reset * self.C
            
            # 进行第二阶段求解
            second_solver = simplex(self.C, self.A, self.B, x0_reset,base_x_index,z0_reset)
            X,Z, _ = second_solver._solve()
            return X,Z
        else:
            return None, None
        
        
    
    def solver(self, method):
        self.standard_trans()
        x0,x0_index,z0 = self.initial_x()
        if x0 != None:
            # 采用”单纯形“法进行计算
            simplex_solver = simplex(self.C, self.A, self.B, x0,x0_index,z0)
            X, Z, _ = simplex_solver._solve()
            return X, Z
        else:
            # 采用”两阶段“法进行求解
            return self._twoPhase()
        