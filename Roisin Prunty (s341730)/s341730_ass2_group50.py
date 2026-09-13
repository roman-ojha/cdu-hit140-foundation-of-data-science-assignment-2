import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
import numpy as np
import scipy.stats as st
import math
import seaborn as sns

dfFifa = pd.read_csv('Fifa2026/Fifa_Pop.csv')
#print(dfFifa.info())


dfFifa['Goal_diff_as_absolute'] = np.sqrt(dfFifa['Goal_diff']*dfFifa['Goal_diff'])

dfFifa['win_draw_lose'] = np.where(dfFifa['Goal_diff']==0,0,
                                   np.where(dfFifa['Goal_diff']>0,1,-1))

dfFifa = dfFifa.iloc[:,[1,2,3,4,-2,-1]]


sampleFifa = dfFifa.sample(n=40,random_state=23)


print('Fifa sample dataframe:\n',sampleFifa.info())
print('Summary Statistics of Fifa sample dataframe:\n',sampleFifa.describe())



df1 = sampleFifa['Total_foul']

sns.boxplot(y=df1)
plt.title('Boxplot of Total Fouls per match in Fifa 2026 sample')
plt.show()

#remove outliers - IQR*1.5

q1= np.percentile(df1, 25)
q3 = np.percentile(df1, 75)
iqr = q3-q1

low_limit = q1- (iqr*1.5)
up_limit = q3+ (iqr*1.5)

no_outliers = ((df1 >= low_limit) & (df1<= up_limit))
sampleFifa_no_outlier = sampleFifa[no_outliers]
df_outliers = df1[(df1<low_limit)|(df1 > up_limit)]
print('Outliers removed: ',len(df_outliers) )
print('The values being:', df_outliers.values)
print(sampleFifa_no_outlier.info())

print('Number of outliers removed: ',len(df_outliers),'\nThe values being:', df_outliers.values)



skew = sampleFifa_no_outlier['Total_foul'].skew()
plt.hist(sampleFifa_no_outlier['Total_foul'])
plt.title('Distribution of Fouls per match,\nFifa 2026 sample\nOutliers removed [24,25,24]')
plt.xlabel('Fouls')
plt.ylabel('Matches')
plt.text(x=5.5,
         y=8,
         s="Skew= %.2f"%skew)

plt.show()
sns.pairplot(sampleFifa_no_outlier, hue='win_draw_lose')
plt.show()

plt.scatter(sampleFifa_no_outlier['Total_foul'],sampleFifa_no_outlier['Goal_diff_as_absolute'])
plt.title('Plot of Fouls to Goal difference,\nFifa 2026 sample')
plt.xlabel('Fouls')
plt.ylabel('Goal difference')

plt.show()

plt.scatter(sampleFifa_no_outlier['Total_foul'],sampleFifa_no_outlier['win_draw_lose'])
plt.title('Plot of Fouls to Win/Draw/Loss status,\nFifa 2026 sample')
plt.xlabel('Fouls')
plt.ylabel('Win/Draw/Loss')


plt.show()


var1 = sampleFifa_no_outlier['Total_foul']
var2 = sampleFifa_no_outlier['win_draw_lose']

corr1,p1 = st.pearsonr(var1, var2)
corr2,p2 = st.spearmanr(var1,var2)

print("Test for correlation between variables Total_Fouls(outliers removed) and Win/Draw/Lose \n",
      "using Pearson's R for linear relationship and Spearmans Rank for non-linear relationship" )
print("Pearson's R correlation coefficient: %.2f,\n p-value: %.2f"%(corr1,p1))
print('Spearmans Rank correlation coefficient: %.2f,\n p-value: %.2f'%(corr2,p2))



#CI
x_bar = stats.mean(sampleFifa_no_outlier['Total_foul'])
s = sampleFifa_no_outlier['Total_foul'].std(ddof=1)
n = (len(sampleFifa_no_outlier['Total_foul']))

z_score= st.norm.ppf(q=.975)
std_err = s/math.sqrt(n)
mrg_err = z_score*std_err

print("Mean:    %.2f\n Standard deviation:  %.2f\n Sample Size:   %d" % (x_bar, s, n))

print('Z-statistic at 95percent confidence: %.2f\n Standard Error:  %.2f\n Margin of Error: %.2f'%(z_score,std_err,mrg_err))
print('The mean number of Fouls in a match during Fifa2026 is %.2f +/- %.2f or, within the range of %.2f to %.2f'%(x_bar,mrg_err,x_bar-mrg_err,x_bar+mrg_err))



## comparative statistics
#dfT of matches won or lost & dfD of matches drawn, sample, remove outlier

sns.pairplot(sampleFifa_no_outlier, hue='win_draw_lose')
plt.show()

#dfD dataframe of matches drawn
#Variable 1
dfD = dfFifa[dfFifa['win_draw_lose']==0]
sampleD = dfD.sample(n=40,random_state=23)
sns.boxplot(sampleD['Total_foul'],color='cornflowerblue')
plt.title('Boxplot of Fouls per match,\nFifa 2026 sample of matches who drew')
plt.show()
skew=sampleD['Total_foul'].skew()
plt.hist(sampleD['Total_foul'],color='cornflowerblue')
plt.title('Distribution of Fouls per match,\nFifa 2026 sample of matches who drew')
plt.xlabel('Fouls')
plt.ylabel('Matches')
plt.text(x=15,
         y=8,
         s="Skew= %.2f"%skew)
plt.show()

#dfT dataframe of matches won or lost
# Variable 2
dfT = dfFifa[dfFifa['win_draw_lose']!=0]
sampleT = dfT.sample(n=40,random_state=23)
sns.boxplot(sampleT['Total_foul'], color='darkorange')
plt.title('Boxplot of Fouls per match,\nFifa 2026 sample of matches either Won or Lost')
plt.show()
skew=sampleT['Total_foul'].skew()
plt.hist(sampleT['Total_foul'],color='darkorange')
plt.title('Distribution of Fouls per match,\nFifa 2026 sample of matches either Won or Lost')
plt.xlabel('Fouls')
plt.ylabel('Matches')
plt.text(x=15,
         y=6,
         s="Skew= %.2f"%skew)
plt.show()


df1= sampleT['Total_foul']
q1= np.percentile(df1, 25)
q3 = np.percentile(df1, 75)
iqr = q3-q1

low_limit = q1- (iqr*1.5)
up_limit = q3+ (iqr*1.5)

no_outliers = ((df1 >= low_limit) & (df1<= up_limit))
sampleT_no_outlier = sampleT[no_outliers]

df_outliers = df1[(df1<low_limit)|(df1 > up_limit)]
print('Variable 2 - Outliers removed: ',len(df_outliers),'Values removed', df_outliers.values)
print(sampleT_no_outlier.info())

skew = sampleT_no_outlier['Total_foul'].skew()
plt.hist(sampleT_no_outlier['Total_foul'], color='darkorange')
plt.title('Distribution of Fouls per match,\nFifa 2026 sample of matches either Won or Lost\nOutliers removed :18 2 18 25 23')
plt.xlabel('Fouls')
plt.ylabel('Matches')
plt.text(x=5.5,
         y=5,
         s="Skew= %.2f"%skew)

plt.show()


#tstat
df1 = sampleD['Total_foul']
x_bar1 = df1.mean()
s1 = df1.std()
n1 = len(df1)

df2 = sampleT_no_outlier['Total_foul']
x_bar2 = df2.mean()
s2 = df2.std()
n2 = len(df2)
print('Base statistics of Variable 1 and Variable 2,\n Variable 1: \nmean- %.2f,\n standard deviation- %.2f,\nsample size- %d'%(x_bar1,s1,n1),
      '\nVariable 2:\nmean- %.2f,\nstandard deviation- %.2f,\nsample size- %d'%(x_bar2,s2,n2))


t_stats, p_val = st.ttest_ind_from_stats(mean1=x_bar1,
                                         std1=s1,
                                         nobs1=n1,
                                         mean2=x_bar2,
                                         std2=s2,
                                         nobs2=n2,
                                         equal_var=False,
                                         alternative="two-sided")


print("\t t-statistic (t*): %.2f" % t_stats)


print("\t p-value: %.3f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    print('P-value < 0.05,\t We reject the null hypothesis.')
else:
    print('P-value > 0.05,\t We accept the null hypothesis.')
