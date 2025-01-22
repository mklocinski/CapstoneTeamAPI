# Overview


# Contents
[Project Summary](#project-summary)

# Project Summary
## Project Description
The goal of the XRAI project was to develop a system that implements explainable AI (XAI) and responsible AI (RAI) principles on a deep reinforcement learning model that is designed to control drone swarms.

## Assumptions
This system was built around the needs of a generic drone data analyst. Given the lack of an actual drone data analyst to interview and test the system, several assumptions were made:
1.	The drone data analyst has a basic understanding of drone swarm operations. 
2.	The drone data analyst has a basic understanding of deep reinforcement learning. The system’s visualizations and data models are based on the basic components of a reinforcement learning model – environment, agents, actions, rewards, and policies – and the drone data analyst is assumed to have a basic understanding of what these components are.  

## Architecture
The XRAI System is designed to allow users to easily execute and analyze the behaviors of the DRL model, which controls a drone swarm’s movements during a mission. The XRAI System provides the following general functionalities:
- **Parameter Selection:** Users can set parameters for each mission, including environment obstacles, RAI expectations, drone swarm settings, and DRL parameters. 
- **Model Execution:** Users can execute the DRL according to their selected parameters and may pause/play at will to analyze mission progress. 
- **ChatGPT Assistant Integration:** Users can send mission-related data and queries to a ChatGPT assistant to explain the drone swarm’s behavior. 
- **Visualizations:** Users can view visualizations of the DRL’s output to further enhance their understanding of the drone swarm’s behavior. 
The system is comprised of a web-based UI and API that manages DRL execution and database operations required to store and share DRL output. 

### High-Level Architecture Diagram 
![Flow chart depicting the architecture of the XRAI System](/assets/app_architecture.png)

### Data Flow: User, UI, API
![Flow chart depicting the flow of data from the user, to the UI, through to the API](/assets/app_data_flow.png)


```mermaid
sequenceDiagram
  participant User
    participant UI
    participant API
    participant Model Thread
    participant Model Run
    participant Database
    participant OpenAI

    User->>+UI: Enter system parameters
    activate UI
    User->>UI: Click "Run Model"
    UI->>API: POST<br/> model/standard/run_xrai <br/>payload: system parameters
    activate API
    API->>API: routes.set_up_flag_files()<br/> returns model_status.txt="running", <br/>current_episode.txt="0"
    activate API
    API->>API: routes.make_environment_map(map_parameters) <br/> returns obstacle_df
    activate API
    API->>Model Thread: <<routes.start_model_thread(obstacle_df, system parameters)>>
    activate Model Thread
    API->>API: routes.play_model() <br/>return model_status.txt="running"
    Model Thread->>Model Thread: routes.run_model(obstacle_df, system_parameters)
    Model Thread->>Model Run: <<xrai_runfile.main(obstacle_df, system_parameters)>>
    activate Model Run
    loop for each run
        Model Run->>Model Run: WrappedModel.main(obstacle_df, system_parameters)
        Model Run->>API: WrappedModel.update_episode(), return updated current_episode.txt
        Model Run->>API: WrappedModel.update_db() return model_output.pkl
        Model Run->>API: WrappedModel.check_status() return model_status.txt
        API-->>Model Run: isPaused: check_status()
        alt isPaused: 
            Model Run->>Model Run: time.sleep(1)<br/>WrappedModel.check_status()
        else continue loop       
        end
    end
    User->>UI: Click "Pause"
    UI->>API: POST<br/> model/pause <br/>payload: "pause"
    activate API
        API->>API: routes.pause_model() <br/>return model_status.txt="pause"
    User->>UI: Click "Play"
    UI->>API: POST<br/> model/play <br/>payload: "play"
    activate API
    loop for every other second in run  
        UI->>API: GET /model/current_episode
        API->>API: routes.get_current_episode()<br/>return episode
        API-->>UI: Current Episode
        UI->>API: GET /model/current_status
        API->>API: routes.get_current_status()<br/>return status
        API-->>UI: Current Status
        UI->>API: POST /database/commit
        API->>Database: routes.batched_insertion()
    end
    User->>UI: Assistant Query
    UI->>OpenAI: chat_with_assistant.ask_assistant(query)
    OpenAI->>UI: Assistant Reponse
    UI->>User: Assistant Response
    User->>UI: Assistant Query, <br/> Dataset selected
    UI->>Database: GET /database/last_run/{dataset selection}
    Database->>UI: Query results
    UI->>OpenAI: chat_with_assistant.ask_assistant(query, attachment)
    OpenAI->>UI: Assistant Reponse
    UI->>User: Assistant Response
```
