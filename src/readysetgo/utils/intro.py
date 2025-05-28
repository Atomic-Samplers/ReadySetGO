import pkg_resources
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
    version=pkg_resources.get_distribution('readysetgo').version
    
    return f"""{logo}

{authors}

ReadySetGO version {version}
"""