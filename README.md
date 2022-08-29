# Electric Fence Fault Detector and Localiser

Electric fences are used to establish protected areas (PAs), in agriculture, installations requiring enhanced security like military installations and homes. These fences are prone to faults that affect their effectiveness. Fences used in PAs, especially, are very long. Manual inspection of these fences is labor intensive, time consuming and inefficient. In this project, we have developed a cheap device to remotely monitor electric fences and help in detecting and localising faults in them. The device employs the concept of **Time Domain Reflectometry (TDR)**. TDR involves sending a pulse down a cable and analysing the reflected signal to detect faults, determine the type of faults and their point of occurrence. The device is based on the Raspberry Pi single board computer. The main components that comprise the system are:
1. The Raspberry Pi
2. An ultrafast analogue to digital converter (ADC)
3. A pulse (a Schmitt trigger pulse) generator

A TDR Electric Fence Fault Detector and Localiser has been fabricated on a printed circuit board as shown in Figure 1 below.


<p align="center">
  <img width="600" height="250" src="./images/labeled-tdr-system.jpg"> 
</p>

<p align="center"> 
  <em>Figure 1: A TDR Electric Fence Fault Detector and Localiser</em>
</p>

A section of the [Dedan Kimathi University of Technology](https://www.dkut.ac.ke/) Conservancy's electric fence was used to simulate differrent faults (open circuit and short circuits). Using the TDR system, a square wave was applied to the fence and sampled at the input port. The sampled signals were saved for analysis. The [signals-visualisation](https://github.com/DeKUT-DSAIL/electric-fence-fault-detector-and-localiser/tree/main/signals-visualisation) directory contains a notebook that visualises the signals. The [tdr](https://github.com/DeKUT-DSAIL/electric-fence-fault-detector-and-localiser/tree/main/tdr) directory contains notebooks that uses a simple method of change point detection using numerical derivative to obtain time delay between incident and reflected signals. The distance to the point of the simulated fault from the input port was computed using the obtained time delay.

<p align="center">
  <img width="600" height="250" src="./images/tdr-system-adapter-box.jpg"> 
</p>

<p align="center"> 
  <em>Figure 2: TDR system in an adapter box</em>
</p>


<p align="center">
  <img width="600" height="250" src="./images/tdr-system-connected-to-fence.jpg"> 
</p>

<p align="center"> 
  <em>Figure 3: TDR system connected to an electric fence</em>
</p>


<p align="center">
  <img width="600" height="250" src="./images/short-circuit.jpg"> 
</p>

<p align="center"> 
  <em>Figure 4: A simulation of a short circuit</em>
</p>
