# RAG System for Master Thesis: "Identifying, Mitigating, and Attacking Risks in Retrieval Augmented Generation"

This repository contains code created for the master thesis titled: "Identifying, Mitigating and Attacking Risks in Retrieval Augmented Generation". The system features a comprehensive RAG (Retrieval Augmented Generation) setup, with multiple mitigation techniques designed to handle risks such as data leaks and knowledge poisoning. Additionally, the repository introduces two novel attacks that target these risks.

## File Overview

### Main File
- **RAG.py**: The main file that can be used to recreate the RAG system outlined in this thesis, as well as the two novel attacks. 

### Folders and Their Contents

- **anomaly_detector**  
  This folder contains a trained anomaly detection model based on LOF (Local Outlier Factor). The model is designed to detect and remove most poison attempts from the system.  
  Inside the folder `succes`, you'll find three anomaly detection models that were successfully poisoned using our novel gradual knowledge poison attack.

- **basic_maildir**  
  Contains emails used for the RAG system's database, specifically for the 'basic' role. We use RBAC (Role-Based Access Control) to manage database access.

- **basic_vectorstore**  
  This folder contains the vector store associated with the `basic_maildir` folder.

- **declined_mails**  
  When a new email is added to the RAG system and is declined (due to containing prompt injection or knowledge poison attempts), it is saved here. The status of such emails is first set to 'pending' until an admin user reviews and either accepts or rejects the email using the 'review mails' command.

- **helper_scripts**  
  Contains multiple scripts used to generate results and perform tasks for the thesis.

- **logs**  
  Stores any log files created during the execution of the system.

- **maildir**  
  The full mail directory for the roles 'advanced' and 'admin'.

- **results**  
  This folder contains the results from both novel attacks. The files `fool_lof_xx` show results on the gradual knowledge poison attack, while the files `steal_context_xx` show results on a data leak attempt.

- **prompts.json**  
  This json file was used to carry out the gradual knowledge poison attack.

- **visualization.py**  
  This file was used to generate visuals from the gradual knowledge poison attack for the thesis. We did not store this file in the folder `helper_scripts` as this file relied on all other components of the system.

## Requirements

In order to run the main RAG.py file, you need to install the dependencies listed in the `requirements.txt` file, as well as install `llama3.1` using ollama.
