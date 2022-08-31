
import logging, sys
logging.basicConfig(format="- %(message)s", filemode='w', stream=sys.stdout)
logger = logging.getLogger()

def get_notebook_logger():
    import logging, sys
    logging.basicConfig(format=" - %(message)s", filemode='w', stream=sys.stdout)
    logger = logging.getLogger()
    
    return logger

    

def change_logging_level(logginglevel: str):

    eval(f'logging.getLogger().setLevel(logging.{logginglevel})')

def create_period_list(step: int, end:int,  start:int = 0):
    '''Creates a list of tuples between start and end, with step 'step'
    
    Reason
    -------
    This is used in 07_exploring_consecutive_metrics_all_models_(nb_none) for
    getting the different period in time to calculate the percent of points
    unstable
    Example
    --------
    create_period_list(step = 25, end = 2, start = 0) 
    >> [(0, 24), (25, 49)]
    
    '''
    return [(i * step, (i+1) * step - 1) for i in range(start,end)]



def convert_period_string(period):
    '''
    Converts the periods created by create_period_list to a string.
    Reason
    ------
    This is used in 07_exploring_consecutive_metrics_all_models_(nb_none) 
    
    '''
    period_list = period.split('_')
    return f"Years {int(period_list[0]) + 1} to {int(period_list[1]) + 1}"


def pprint_list_string(l: list, num_start_items=2, no_end_items=0):
    '''This version is useful for logging moduel'''
    
    to_print = f'lenght = {len(l)}'
    for i in range(num_start_items):
        to_print += f'\n {i}. {str(l[i])}'
        
    if no_end_items:
        to_print += '\n...\n'
    for j in range(1, no_end_items+1):
        to_print += f'\n {-j}. {str(l[-j])}'
        
    return to_print


def pprint_list(*args, **kwargs) -> None:
    '''A nicer print of a list with more information'''

    print(pprint_list_string(*args, **kwargs))