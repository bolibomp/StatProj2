import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma, polygamma
from scipy.stats import beta as betadist
import statistics
from scipy.stats import chi2

#--------Packing up the data--------------
data_path = 'RV.csv'
with open(data_path, 'r') as f:
    
    reader = csv.reader(f, delimiter=',')
    
    headers = next(reader)
    
    data = list(reader)
    
    data = np.array(data)
#-----------------------------------------

#-------------Getting the right rows------
ecc_index = headers.index('ECC')

per_index = headers.index('PER')

raw_ecc_data = data[:,ecc_index]

raw_per_data = data[:,per_index]
#-----------------------------------------

#-----Converting and putting in the data in to lists-
ecc_list=[]
for i in raw_ecc_data:
    try:
        ecc_list.append(float(i))
    except ValueError:
        pass

for i in range(len(ecc_list)):
    if ecc_list[i] == 0.0:
        ecc_list[i]=0.0001 #if the value is zero substitute it with 0.0001

per_list=[]
for i in raw_per_data:
    try:
        per_list.append(float(i))
    except ValueError:
        pass
#-----------------------------------------
        
#---making a matrix of the period,eccentricity pairs---     
pair=[[val,per_list[idx]] for idx,val in enumerate(ecc_list)]
pair=np.asarray(pair)
pair=pair[pair[:,1].argsort()]
#-----------------------------------------

#--------Choosing the interval or periods and picking out the corresponding eccentricity----------------
def interval(pair,start,lenght):
    data=[]
    period=[]
    for i in pair:
        if i[1]>=start:
            data.append(i[0])
            period.append(i[1])
        if len(data)==lenght:
            break
    if len(data)!=lenght:
        print('Data size is not the desired one. Data size is now', lenght-len(data),' Change starting value to get the desired data size')
    return np.asarray(data),period
#-----------------------------------------
    
#---Functions for the Newton-Raphson algorithm---
def Jacobian(theta_vector):
    a=theta_vector[0]
    b=theta_vector[1]
    J=[[polygamma(1,a)-polygamma(1,a+b),-polygamma(1,a+b)],[-polygamma(1,a+b),polygamma(1,b)-polygamma(1,a+b)]]
    return np.asarray(J)
           
def vector_of_normal_equations(theta_vector,G_1,G_2):
    a=theta_vector[0]
    b=theta_vector[1]
    V=[digamma(a)-digamma(a+b)-G_1,digamma(b)-digamma(a+b)-G_2]
    return np.asarray(V)
#-----------------------------------------

#-------------Newton-Raphson algorithm. Estimates the parameters----  
def newton_raphson(init_alpha,init_beta,data,tol,max_ite=1e5):
    theta=[init_alpha,init_beta]
    data=np.asarray(data)
    X=1
    count=0
    G_1=1/len(data)*sum(np.log(data))
    G_2=1/len(data)*sum(np.log(1-data))
    while X>tol: #Keeps iterating while lower than some tolerance        
        dtheta = np.linalg.solve(Jacobian(theta),vector_of_normal_equations(theta,G_1,G_2))
        theta = theta -dtheta
        X=np.linalg.norm(dtheta)
        count +=1
        if count>max_ite: #If it keeps iterating to much it breaks
            print('To many iterations')
            break
    return theta, count
#-----------------------------------------
    
#-------------Moments Estimators----------- 
def Moments_Estimators(data):
    mean=1/len(data)*sum(data)
    var=statistics.variance(data)
    alpha=mean*((mean*(1-mean))/(var)-1)
    beta=(1-mean)*((mean*(1-mean))/(var)-1)
    return alpha,beta
#-----------------------------------------

#---------Beta log-liklihood funstion--------   
    
def beta_log(data,X,Y):
    G_1=sum(np.log(data))
    G_2=sum(np.log(1-data))
    return len(data)*np.log(gamma(X+Y))-len(data)*np.log(gamma(X))-len(data)*np.log(gamma(Y))+(X-1)*G_1+(Y-1)*G_2
#------------------------------------------------------------

data=interval(pair,600,50)[0] #Get the eccentricities from the given interval
interval=interval(pair,600,200)[1] #get the interval

print('The calculated values for alpha and beta is:',newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0])# get the parameters
#gets the estimated parameters with the newton rapshon algorithm with the initial guess form the Moments Estimators method and pluggs it in to a beta distribution
dist = betadist(newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][0], newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][1])
#-------------Plot stuff-------------------------------
x = np.linspace(0, 1, 102)
plt.plot(x, dist.pdf(x), c='red', label='Beta distribution') #plot the beta distribtuon with the estimated parameters
plt.hist(data,bins=10,normed=True) #bins the eccentricity data and makes a histogram
plt.xlabel('Eccentricities', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.ylim(0,6)
plt.legend(fontsize=14)
plt.title('Normalized histogram for the eccentricities',fontsize=14)
plt.figure(figsize=(6,6))
plt.show()

delta = 0.025
xrange = np.arange(0.01, 5, delta)
yrange = np.arange(0.01, 5, delta)
X, Y = np.meshgrid(xrange,yrange)
#makes the confidence region from the meshgrod of X,Y values
Z=beta_log(data,X,Y)-beta_log(data,newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][0],newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][1])+2*chi2.ppf(0.9, df=2)
#makes the contour map of the beta log-liklihood from the meshgrod of X,Y values
W=beta_log(data,X,Y)


plt.contour(X, Y, Z,levels=[1])
plt.contourf(X, Y, W,1000)
plt.colorbar().set_label(r'$\log(L)$',fontsize=18)
plt.ylim(0,5)
plt.xlim(0,5)
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
alpha=newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][0]
beta=newton_raphson(Moments_Estimators(data)[0],Moments_Estimators(data)[1],data,1e-10)[0][1]
plt.plot([alpha],[beta],'+',markersize=10)
plt.text(0.45, 0.93, r'$\hat\alpha\approx$0.57$ \hat\beta\approx$2.0', bbox=dict(ec='k', fc='w', alpha=0.9), ha='center', va='center', transform=plt.gca().transAxes)
plt.title(str(r'$\Delta$Period from ')+str(min(interval).round(1))+str(' to ')+str(max(interval).round(1))+str(' with size 200'))
plt.figure(figsize=(6,6))
plt.show()