# Adaptive Mechanisms to Refund Emissions Payments
Code in this repository implements an agent-based model to investigate the design of an adaptive recalibration strategy to refund emissions payments, and is principally based on the concept of a Refunded Emissions Payment (REP) scheme. This model is implemented in the context of a power system, and uses data describing Australia's largest electricity transmission network, which are obtained from the following link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942).

A unit commitment model is used to model the Australian National Electricity Market at hourly resolution. The scheme seeks to augment the short-run marginal costs of generators by imposing a net liability per MWh on generators under the scheme's remit of the following form:

net liability per MWh = ![(E_{g} - \phi)\tau](https://render.githubusercontent.com/render/math?math=(E_%7Bg%7D%20-%20%5Cphi)%5Ctau)

Where:

![E_{g}](https://render.githubusercontent.com/render/math?math=E_%7Bg%7D) = emissions intensity of generator ![g](https://render.githubusercontent.com/render/math?math=g) (tCO2/MWh)
 
![\phi](https://render.githubusercontent.com/render/math?math=%5Cphi) = emissions intensity baseline (tCO2/MWh)

![\tau](https://render.githubusercontent.com/render/math?math=%5Ctau) = permit price ($/tCO2)

The form of this net liability is very similar to that of a Tradable Performance Standard (TPS) or Emissions Intensity Scheme (EIS). However, unlike these mechanisms the policymaker in assumed to have control over both the permit price and the baseline. Under a TPS or EIS the baseline is fixed with demand and supply determining the permit price. This is in contrast to a REP scheme where the permit price is fixed, but the baseline is uncertain (the baseline is equal to the average emissions intensity of generators under the scheme's remit). By fixing both the baseline and the permit price generators have greater certainty with respect to the net liability they face. However, the cost of doing so is uncertainty with respect to net accrued scheme revenue as the scheme's are no longer revenue neutral (net penalties received may not exactly equal net credits paid-out). This uncertainty is managed via a model predicitive control framework that periodically updates the baseline with the objective of attaining a revenue target defined by the policymaker. 

Code within this repository collates and processes data, and also runs agent-based simulations in order to examine properties of the presented scheme. 

## Addendum
There is a nice duality between TPS and REP schemes. Under a REP scheme the permit price is fixed, with the net liability faced per MWh taking the same form as is presented above. However, the baseline in this instance is the average emissions intensity of energy produced by all generators under the scheme's remit. Consequently, the net liability per MWh is only known after energy has been delivered. Conversely, under a TPS the benchmark rate is fixed, but some uncertainty remains with respect to the permit price (which is determined by the demand and supply of permits). To summarise, under a REP scheme the permit price is fixed but the baseline is uncertainty, and under a TPS the baseline is fixed but the permit price is uncertain. The presented framework fixes both the baseline and the permit price. The cost of doing so is uncertainty with respect to scheme revenue (net penalties received may not exactly equal net credits paid-out) and the scheme is no longer revenue-neutral. The analysis suggests that this uncertainty with respect to scheme revenue can be managed by incrementally changing the baseline using a model predictive controller, with the framework also providing the ability to flexibly target auxiliary objectives.