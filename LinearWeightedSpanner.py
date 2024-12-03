import numpy as np
from LW_ArgMax import LW_ArgMax


def LWS(params , arm_set , eta , best_arm , theta):
    """
    returns a C/alpha-approximate barycentric spanner assuming LW-ArgMax
    returns a alpha-approimate solution with multiplicative error
    """
    barycentric_spanner = [arm for arm in np.identity(params["dimension"])]

    def weigh_the_arm(arm):
        gap = np.dot(best_arm , theta) - np.dot(arm , theta)
        return np.array(arm) / (1 + eta * gap)

    A = [arm for arm in np.identity(params["dimension"])]
    C = np.exp(1)

    phi_arms = [weigh_the_arm(a) for a in arm_set]

    for i in range(params["dimension"]):
        e_vec = [0 for _ in range(params["dimension"])]
        e_vec[i] = 1
        try:
            estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
        except:
            print("Error in A", A)
            break
        a_plus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta)
        a_minus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , -theta)
        if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
            A[i] = weigh_the_arm(a_plus)
            barycentric_spanner[i] = a_plus
        else:
            A[i] = weigh_the_arm(a_minus)
            barycentric_spanner[i] = a_minus

    print("Loop 1 of LWS done")

    replacement = True
    while replacement:
        replacement = False
        for i in range(params["dimension"]):
            e_vec = [0 for _ in range(params["dimension"])]
            e_vec[i] = 1
            
            try:
                estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
            except:
                print("Error in A", A)
                break

            a_plus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta)
            a_minus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , -theta)

            if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
                a = weigh_the_arm(a_plus)
            else:
                a = weigh_the_arm(a_minus)

            A_temp = A.copy()
            A_temp[i] = weigh_the_arm(a)
            if np.linalg.det(A_temp) >= C * np.linalg.det(A):
                print("Found a better candidate. Restarting Loop")
                A[i] = weigh_the_arm(a)
                barycentric_spanner[i] = a
                replacement = True
                break
    print("Loop 2 of LWS done")            
    return barycentric_spanner