from importlib.metadata import version


def create_intro():
    logo="""             XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX             
         XXXX                                                   XXXX         
      XXX    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    XXX      
    XX   XXXX                                                   XXXX   XX    
   X   XX    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    XX   X   
  X   X   XX                                                    XXX   X   X  
 X  XX  XX   XXXXXXXX        XXXXXXX     XXXXXXX       XXXXXX      XX  XX  X 
X   X  X     XX      XX    XXX          XX           XX      XX      X  X   X
X  X  X      XX      XX    XX           XX          XX        XX      X  X  X
X  X  X      XXXXXXXXX      XXXXXXXX    XX   XXXX   XX        XX      X  X  X
X  X  X      XX     XX             XX   XX     XX   XX        XX      X  X  X
X   X  X     XX      XX    XXX    XXX   XXX    XX    XX      XX      X  X   X
 X  XX  XX   XX       XX     XXXXXXX      XXXXX        XXXXXX      XX  XX  X 
  X   X   XXX                                                   XXX   X   X  
   X   XX    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    XX   X   
    XX   XXXX                                                   XXXX   XX    
      XXX    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    XXX      
         XXXX                                                   XXXX         
             XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX             """
    authors="Dr. Julian Holland"
    rsgo_version=version('readysetgo')
    
    return f"""{logo}

{authors}

ReadySetGO version {rsgo_version}
"""