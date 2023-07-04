# GLAM Logit: A random-utility-consistent method to estimate non-parametric coefficents from ubiquitous datasets

## 1. Introduction

Group Level Agent-based Mixed (GLAM) Logit is a variation of mixed logit (MXL) model, which provides deterministic and agent-specific estimation that can be efficiently integrated into optimization models. This is an extension of agent-based mixed logit (AMXL) model in Ren and Chow (2022)’s study. Consider within each agent $i ∈ I$ , there is a population $N$ of individuals behaviorally seek to select to maximize their overall utilities. In line with the random utility theory, the total utility derived from an individual $n ∈ N$ choosing alternative $j$ is defined as:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Eq1.JPG" height="40px">

where $U_{nj}$ is the total utility, which is composed of a deterministic utility $V_{nj}$ and a Gumbel-distributed random utility $\epsilon_{nj}$. $X_{nj}$ is a set of observed variables related to individual n choosing alternative $j$. $\theta_i$ is a vector of taste coefficients of individuals in agent $i$. By doing this, we capture individual heterogeneities within agent $i$ with $\epsilon_{nj}$ and we assume individuals in the same agent share the same set of coefficients $\theta_i$. According to Berry et.al (1995), the relationship between the utility function and the choice probability can be defined as:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Eq2.JPG" height="60px">

where $S_{ij}$ is the market share of alternative $j$ in agent $i$; $ln⁡(S_{ij})-ln⁡(S_{ij*})$ is called inverted market share, which can be measured as observed variables $X_{nj}$, $X_{nj*}$ and coefficients $\theta_i$. The agent-level coefficient set $\theta_i$ can be estimated by solving a multiagent inverse utility maximization (MIUM) problem under $L_2$-norm as a convex quadratic programming (QP) problem, as illustrated in the following equations:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Eqs3.JPG" width="500px">

where $\theta^k_0$ is the $k^{th}$ fixed-point prior corresponding to a latent class; $tol$ is a manually set tolerance to draw a balance of goodness-of-fit and feasible solutions (a larger value of tol leads to a lower goodness-of fit while a higher proportion of feasible solutions). A recommend range of tol is [0.1,1.5]. The estimated agent coefficients have a consistency with one of the fixed-point priors. $I^k$ denotes the agent set of the $k^{th}$ latent class, which can be determined by applying the K-Means algorithm to $\theta_i$. The objective is quadratic while the constraints are linear. Solving the problem as a single QP would be computationally costly. Instead, we solved this $|I|$ times with $\theta_i$ as the decision variables in each iteration (which are much smaller QPs). At the end of each iteration, we applied K-Means algorithm to $\theta_i$ to identify latent classes, updated the fixed-point priors, and check if the stopping criteria has been reached. The iterations continue until all priors $\theta^k_0$ stabilize (see Xu et al. (2018) for an example of this kind of decomposition for the $|K|=1$ case). The estimation alogrithm is illustrated as follows, which is solved using python Gurobi package.

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Algorithm illustration.JPG" width="650px">

For more details, please refer to the following paper:

[Ren, X., & Chow, J. Y. (2022). A random-utility-consistent machine learning method to estimate agents’ joint activity scheduling choice from a ubiquitous data set. Transportation Research Part B: Methodological, 166, 396-418.](https://www.sciencedirect.com/science/article/pii/S0191261522001862)

[Xu, S. J., Nourinejad, M., Lai, X., & Chow, J. Y. (2018). Network learning via multiagent inverse transportation problems. Transportation Science, 52(6), 1347-1364.](https://pubsonline.informs.org/doi/abs/10.1287/trsc.2017.0805)

[Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile prices in market equilibrium. Econometrica: Journal of the Econometric Society, 841-890.](https://www.jstor.org/stable/2171802)

## 2. License
The NYU NON-COMMERCIAL RESEARCH LICENSE is applied to EVQUARIUM (attached in the repository). Please contact [Joseph Chow](https://github.com/jc7373) (joseph.chow@nyu.edu) for commercial use.

For questions about the code, please contact: [Xiyuan Ren](https://github.com/xr2006) (xr2006@nyu.edu).

## 3. Examples

### A simple example

We built a simple example of mode choice to illustrate how the GLAM logit works. In this example, each agent refers to trips belonging to an OD pair. Only two modes, taxi and transit, are considered for simplicity. Each row of the sample data contains the ID of the agent, travel time and cost of taxi, travel time and cost of transit, and mode share of the two modes. 

It is noted that we added two “fake” agents (agent 7 and 8) into the dataset. The mode shares of these two agents are unreasonable since the mode with a longer travel time and a higher cost has a higher market share. The sample data containing aggregated-level mode choice information of 8 agents:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Example_Fig1.JPG" width="600px">

The derived utilities of the two modes are defined as:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Example_Fig2.JPG" width="650px">

where $V_{taxi,i}$ and $V_{transit,i}$ are utilities derived from choosing taxi and transit. $\theta_{time,i}$ and $\theta_{cost,i}$ are the coefficients of travel time and cost for agent $i∈I$. $\theta_{c-transit,i}$ is the mode constant for agent $i$. 

We ran GLAM Logit with latent class $K=3$. The estimation results are shown as follows:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Example_Fig3.JPG" width="600px">

The estimated market share E_Taxi (%) and E_Transit (%) are quite close to the input data. Moreover, the results reflect diverse tastes at the agent level though the three latent classes: (1) agent 1-3 have negative $\theta_{time,i}$ and $\theta_{cost,i}$ close to zero, indicating a preference for shorter travel time; (2) agent 4-6 have negative $\theta_{cost,i}$ and $\theta_{time,i}$ close to zero, indicating a preference for lower travel cost; (3) agent 7 and 8 have positive $\theta_{time,i}$ and $\theta_{cost,i}$, indicating an “irregular” preference for longer travel time and higher travel cost. In ubiquitous datasets, “irregular” preference is often related to issues in data collection. To this end, GLAM logit can be used to check the data quality in some cases.

For detailed codes, please check [Illustrative_sample.py](https://github.com/BUILTNYU/GLAM-Logit/blob/main/GLAM-Logit/Illustrative_sample.py)

### New York Statewide mode choice modeling

In a real case study, a NY statewide model choice model is developed using GLAM Logit. Synthetic trips on a typical weekday were used to calibrate the model. We considered six modes enabled by Replica’s datasets, including private auto, public transit, on-demand auto, biking, walking, and carpool. The GLAM logit model with 120,740 agents took 2.79 hours to converge at the 26th iteration, with a rho value of 0.6197.

Coefficient distribution: 

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Example_Fig4.jpg" width="900px">

Value of time (VOT) of different population segments in NYC and NY state:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/VOT.JPG" width="900px">

Value of time (VOT) distribution in NY state and NYC:

<img src="https://github.com/BUILTNYU/GLAM-Logit/blob/main/img_file/Example_Fig5.jpg" width="900px">

## 4. Instruction

To run GLAM Logit model:

Please conduct the following steps: 1) define the utility function; 2) prepare group-level choice observation datasets (see [OD_level_RP_processing.py](https://github.com/BUILTNYU/GLAM-Logit/blob/main/GLAM-Logit/OD_level_RP_processing.py)), 3) run inverse optimization algorithm for a single agent (see [Group_level_IO.py](https://github.com/BUILTNYU/GLAM-Logit/blob/main/GLAM-Logit/Group_level_IO.py)), and; 4) run the whole estimation algorithm (see [Model_building.py](https://github.com/BUILTNYU/GLAM-Logit/blob/main/GLAM-Logit/Model_building.py))

For further questions, please contact: [Xiyuan Ren](https://github.com/xr2006) (xr2006@nyu.edu).

## 5. Significance

Compared with conventional logit models (e.g., MNL, NL, MXL), the significance of GLAM Logit model is three-fold. 

- GLAM Logit takes OD level (instead of individual level) data as inputs, which is efficient in dealing with ubiquitous datasets containing millions of observations. 
- Preference heterogeneities are based on non-parametric aggregation of coefficients per agent instead of having to assume a distributional fit. The spatial distribution of agent-level coefficients is infeasible for conventional logit models to capture.
- GLAM Logit can be directly integrated into optimization models as constraints instead of dealing with simulation-based approaches required by mixed logit (MXL) models. For instance, multi-service region assortment can be formulated as a quadratic programming (QP) problem. 
