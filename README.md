# Electric Fence Fault Detector and Localiser

Electric fences are used to establish protected areas (PAs), in agriculture, installations requiring enhanced security like military installations and homes. These fences are prone to faults that affect their effectiveness. Fences used in PAs, especially, are very long. Manual inspection of these fences is labor intensive, time consuming and inefficient. In this project, we are developing a cheap device to remotely monitor electric fences and help in detecting and localising faults in them. The device will employ the concept of **Time Domain Reflectometry (TDR)**. TDR involves sending a pulse down a cable and analysing the reflected signal to detect faults, determine the type of faults and their point of occurrence. The device will be based on the Raspberry Pi single board computer. The main components that comprise the system are:
1. The Raspberry Pi
2. An ultrafast analogue to digital converter (ADC)
3. A pulse (a Schmitt trigger pulse) generator 
