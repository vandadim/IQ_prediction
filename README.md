# IQ prediction

# Intelligence, brain size, and brain morphometry: resolving conflicting claims


Brain size is associated with intelligence, but the relationship between brain morphometric measures and intelligence is unclear. Studies have produced conflicting results or no significant relations between intelligence and morphometric measures such as cortical thickness and peri-cortical contrast. This discrepancy may be due to a failure to fully account for the relationship between brain size and intelligence. Our study shows that neither cortical thickness nor peri-cortical contrast improves IQ prediction accuracy beyond what is achieved with brain volume alone.

**TABLE OF CONTENTS**
===================================
[1. Introduction](#1-introduction)

[2. Dataset](#2-Dataset)
  + [2.1 ABCD Data](#21-ABCD)    
  + [2.2 NIHPD Data](#22-NIHPD)
  + [2.3 NKI-RS Data](#23-NKI-RS)
  
[3. Methods](#3-Methods)
  + [3.1 Preprocessing](#31-Preprocessing)
  + [3.2 Regression models](#32-Models)    
    + [3.2.1 Elastic net](#321-GLMNET)    
    + [3.2.2 Random forest](#322-RF)    
    + [3.2.3 SVR](#323-SVR)    
    + [3.2.1 XGBoost](#321-XGBoost)


# INTRODUCTION
Alzheimer’s Disease (AD) is a chronic neurodegenerative disorder that occurs among the elderly. AD's pathophysiological changes begin many years before clinical manifestations of disease and the spectrum of AD spans from clinically asymptomatic to severely impaired. Because of this, there is an appreciation that AD should not only be viewed with discrete and defined clinical stages but as a multifaceted process moving along a continuum. Therefore, early prediction of disease progression would be a crucial step towards designing proper therapeutic, unburden the health care system, and preventing adverse events caused by AD. Due to this reason predicting AD
