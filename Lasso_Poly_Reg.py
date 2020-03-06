def Mariamz_Lasso_Poly_Reg(data_frame,random_state,max_degree_of_polynomial,range_of_alpha,resolution_of_lasso_iteration):
    
    import  numpy as np
    import  pandas as pd
    import  sklearn 
    import matplotlib.pyplot as plt
    from  sklearn.model_selection  import train_test_split 
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LinearRegression
    
    X= data_frame.iloc[:,0:data_frame.shape[1]-1].values
    y= data_frame.iloc[:,data_frame.shape[1]-1].values 
    if data_frame.shape[1]==2:
        X= X.reshape(-1,1)   # RESHAPED
        y=y.reshape(-1,1)
  
    
    #R2_train=[]
    R2_test=[]
    Deg=[]
    Alpha=[]
    R2_test_lasso=[]
    final_degree=[]
    final_alpha=[]
    
    X_train,X_test,y_train , y_test =  train_test_split(X,y,test_size=0.3, random_state=random_state,shuffle=True)
    
    for i in range(max_degree_of_polynomial):
        PL= PolynomialFeatures(degree=i)
        Deg.append(i)
        p_X_train=PL.fit_transform(X_train)
        p_X_test=PL.fit_transform(X_test)
        LR=LinearRegression()
        LR_fit=LR.fit(p_X_train,y_train)
        LR_pred_train=LR_fit.predict(p_X_train)
        LR_pred_test=LR_fit.predict(p_X_test)
        r2_train=r2_score(y_train,LR_pred_train)
        r2_test=r2_score(y_test,LR_pred_test)
        #R2_train.append(r2_train)
        R2_test.append(r2_test)
    for d,r in zip(Deg,R2_test):
        if r==max(R2_test):
            final_degree.append(d)
    
    PL_d= PolynomialFeatures(degree=d)
    p_X_train_d=PL_d.fit_transform(X_train)
    p_X_test_d=PL_d.fit_transform(X_test)
    
    for alpha in np.arange(0,range_of_alpha,resolution_of_lasso_iteration):
        lasso = Lasso(alpha = alpha, normalize=True)
        Alpha.append(alpha)  
        lso_fit=lasso.fit(p_X_train_d, y_train)
        lso_pred_test=lso_fit.predict(p_X_test_d)
        r2_test_lasso=r2_score(y_test,lso_pred_test)
        R2_test_lasso.append(r2_test_lasso)
    for a,r in zip(Alpha,R2_test_lasso):
        if r==max(R2_test_lasso):
            final_alpha.append(a)
    #return final_degree[0], final_alpha[0]
    lasso_final=Lasso(alpha=final_alpha[0],normalize=True)
    lasso_final_fit=lasso_final.fit(p_X_train_d, y_train)
    lasso_final_predict=lasso_final_fit.predict(p_X_test_d)
    lasso_final_predict_r2=r2_score(y_test,lasso_final_predict)
   
    return final_degree[0], final_alpha[0],lasso_final_predict_r2
   
    
    
