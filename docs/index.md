# ![TrainKeeper logo](assets/trainkeeper-logo.png)
# TrainKeeper
 
 TrainKeeper is a minimal-decision, high-signal toolkit for reproducible, debuggable, and efficient training workflows.
 
 ## Quick start
 ```python
 from trainkeeper.experiment import run_reproducible
 
 @run_reproducible(auto_capture_git=True)
 def train():
     ...
 ```
 
 ## Modules
 - experiment
 - datacheck
 - trainutils
 - debugger
 - pkg
 - monitor
