"""
Implementation of a task set
"""
import numpy as np
from utils.proc_env import *
from gym_minigrid.envs import ProceduralEnv
from gym_minigrid.register import register

class TaskSetup(object):
    def __init__(self, task_sets):
        assert len(task_sets) == 2, 'other lengths not supported at this moment'

        self.unique_tasks = set([])
        self.id2task = {}
        for task_set in task_sets:
            if task_set is not None:
                for task in task_set:
                    self.unique_tasks.add(task.id)
                    self.id2task[task.id] = task

        self.idx2taskid = {idx: taskid for idx, taskid in enumerate(list(self.unique_tasks))}
        taskid2idx = {taskid: idx for idx, taskid in enumerate(list(self.unique_tasks))}

        for task_set in task_sets:
            if task_set is not None:
                for task in task_set:
                    encoding = [0]*len(taskid2idx)
                    encoding[taskid2idx[task.id]] = 1
                    task.set_id_encoding(encoding)
        self.base_set = task_sets[0]
        self.transfer_set = task_sets[1]

    @property
    def num_unique_tasks(self):
        return len(self.unique_tasks)

    @property
    def base_task_set(self):
        return self.base_set

    def transfer_task_set(self):
        return self.transfer_set

    def get_task_by_id(self, id_):
        assert id_ in self.id2task, 'Invalid task id'
        return self.id2task[id_]

    def get_taskid_by_idx(self, idx):
        assert idx in self.idx2taskid
        return self.idx2taskid[idx]

class TaskSet(object):
    """
    Holds mappings from task id to symbol string
    Holds mapping from task id to environment
    holds mapping from task id to natural language description of task
    """
    def __init__(self, task_set, vocab):

        # task_tups : (name, environment, symbol string)
        assert isinstance(task_set, list)
        assert all([isinstance(element, Task) for element in task_set])
        self.task_set = task_set
        self.vocab = vocab # this contains the universe of symbols
        # passing the vocab to each of the tasks
        for task in self.task_set:
            task.hook_vocab(vocab)

    @property
    def size(self):
        return len(self.task_set)

    def __getitem__(self, item):
        return self.task_set[item]

    def __str__(self):
        return "\n".join([str(task) for task in self.task_set])

# TODO:
class Task(object):
    def __init__(self, name, environment, symbol_sequence):
        self.name = name
        # this can either be a string or the whole environment object.
        self.env_str = environment

        assert isinstance(symbol_sequence, SymbolSequence)
        self.symbol_seq = symbol_sequence

        # things that need to be initialized from higherlevels of the classes 
        self.symbol_vocab = None
        self.id_encoding = None

    @property
    def symbs(self):
        return self.symbol_seq

    @property
    def env(self):
        return self.env_str

    @property
    def id(self):
        return self.name

    def encode_symbol(self, symb):
        return self.symbol_vocab.encode(symb)

    def __str__(self):
        if isinstance(self.env_str, str):
            s = "{}: {}  |  {}".format(self.name, self.env_str, str(self.symbol_seq))
        else:
            s = "{} | {}".format(self.name, str(self.symbol_seq))
        return s

    def hook_vocab(self, vocab):
        self.symbol_vocab = vocab

    def set_id_encoding(self, encoding):
        self.id_encoding = encoding 

    def encode_self(self):
        assert self.id_encoding is not None, "id_encoding hasn't been initialized. Make sure to make a TaskSetup object."
        return self.id_encoding

class SymbolSequence(object):
    def __init__(self, seq=None):
        if seq is not None:
            assert isinstance(seq, list)
            assert all([isinstance(element, Symbol) for element in seq])
            self.seq = seq
        else:
            self.seq = []

    def add_symbol(self, symb):
        assert isinstance(symb, Symbol)
        self.seq.append(symb)

    def __getitem__(self, item):
        return self.seq[item]

    def __str__(self):
        return ', '.join([str(el) for el in self.seq])

    def __len__(self):
        return len(self.seq)

class Symbol(object):
    def __init__(self, id_, description=None):
        self.id_ = id_
        self.descr = description

    @property
    def description(self):
        return self.descr

    @property
    def id(self):
        return self.id_

    def __str__(self):
        return self.id_

class SymbVocabulary(object):
    def __init__(self, symbols):
        assert isinstance(symbols, list)
        assert all([isinstance(el, Symbol) for el in symbols])
        self.symbols = symbols
        self.symbol2idx = {symbol.id: idx for idx, symbol in enumerate(self.symbols)}


    def decode(self, rep):
        index = rep.index(1)
        return self.symbols[index]

    def encode(self, symbol):
        assert symbol.id in self.symbol2idx, "Invalid symbol"
        rep = [0] * self.size
        rep[self.symbol2idx[symbol.id]] = 1
        return rep

    def __getitem__(self, item):
        return self.symbols[item]

    def __str__(self):
        return "\n".join([str(symb) for symb in self.symbols])

    @property
    def size(self):
        return len(self.symbols)


# The following are utility functiosn to create tasks.

def get_procedural_taskenvs(num_procs, num_tasks_per_length, max_length=5,
                            room_size=6, max_cluster_size=2, seed=None):
    """
    Output
    procedural_env_list is going to be a (max_length x num_tasks_per_length) x num_procs list
    procedural_task_list is going to be a (max_length x num_tasks_per_length) list
    """
    if seed is not None:
        np.random.seed(seed)
    procedural_env_list = []
    procedural_task_list = []
    for seq_length in range(1, max_length+1):  # maximum sequence length of 5
        for count in range(num_tasks_per_length):
            print("creating tasks of length {}".format(seq_length)) 
            # create the sequence, subject to some constraints.
            key_count = 0
            goal_in_sight = False
            new_task_sequence = []
            for i in range(seq_length):
                # if the key count is positive, then you've got all the options
                if key_count > 0:
                    limited_choices = subtasks
                else:
                    # exclude consideration
                    limited_choices = [element for element in subtasks if element.id != 'unlock_correct_door']

                if goal_in_sight:
                    limited_choices = [element for element in limited_choices if element.id != 'to_room_with_goal']
                    chosen_subtask = np.random.choice(limited_choices)
                    goal_in_sight = False  # whatever the choice, you will reset
                else:
                    limited_choices = [element for element in limited_choices if element.id != 'move2goal']
                    chosen_subtask = np.random.choice(limited_choices)
                    if chosen_subtask.id == 'to_room_with_goal':
                        goal_in_sight = True
                if chosen_subtask.id == 'move2key':
                    key_count += 1
                elif chosen_subtask.id == 'unlock_correct_door':
                    key_count -= 1
                new_task_sequence.append(chosen_subtask)

            print(','.join([str(el) for el in new_task_sequence]))

            # create num_procs instantiations of this environment
            this_procedural_env_list = []
            for proc_idx in range(num_procs):
                # create the graph of rooms
                procedural_graph = ProceduralGraph(new_task_sequence, cluster_max_size=max_cluster_size, seed=None if seed is None else proc_idx*seed)
                rowcol, locked, unlocked, hallways = procedural_graph.get_rowcol()
                connections = {'locked': locked, 'unlocked': unlocked, 'hallways': hallways}
                # instantiate the room
                new_prenv = ProceduralEnv(new_task_sequence, connections, rowcol,
                                          procedural_graph.room2pos, procedural_graph.possessions,
                                          procedural_graph.starting_point, room_size=room_size,
                                          seed= None if seed is None else proc_idx*seed)
                # save into global list
                this_procedural_env_list.append(new_prenv)
            procedural_task_list.append(Task('l{}_i{}'.format(seq_length, count), None, SymbolSequence(new_task_sequence)))
            procedural_env_list.append(this_procedural_env_list)

    return procedural_env_list, procedural_task_list

def get_taskenvs_from_strs(strs):
    assert isinstance(strs, list)
    id2subtask = {subtask.id: subtask for subtask in subtasks}
    new_task_sequence = []
    for s in strs:
        assert s in id2subtask
        new_task_sequence.append(id2subtask[s])

    # create the graph of rooms
    procedural_graph = ProceduralGraph(new_task_sequence, cluster_max_size=2)
    rowcol, locked, unlocked, hallways = procedural_graph.get_rowcol()
    connections = {'locked': locked, 'unlocked': unlocked, 'hallways': hallways}
    # instantiate the room
    new_prenv = ProceduralEnv(new_task_sequence, connections, rowcol,
                              procedural_graph.room2pos, procedural_graph.possessions,
                              procedural_graph.starting_point, room_size=6)

    new_task = Task('--'.join(strs), None, SymbolSequence(new_task_sequence))
    # this new task needs to be joined with the vocabulary.
    _ = TaskSet([new_task], vocab)
    _ = TaskSetup([[new_task], None])
    return new_prenv, new_task

def generate_default_task():
    # returns the base and transfer sets.
    task1 = Task('to-room-with-goal', 'MiniGrid-FourRooms-ToRoomWithGoal-v0',
                SymbolSequence([to_room_with_goal]))
    task2 = Task('to-goal', 'MiniGrid-FourRooms-Get2Goal-v0',
                SymbolSequence([move2goal]))
    task3 = Task('4-room', 'MiniGrid-FourRooms-v0',
                SymbolSequence([to_room_with_goal, move2goal]))
    task4 = Task('unlock', 'MiniGrid-Unlock-v0',
                SymbolSequence([move2key, unlockcorrectdoor]))  # this has been changed
    global_taskset = TaskSet([task1, task2, task3, task4], vocab)

    task5 = Task('door-key', 'MiniGrid-DoorKey-5x5-v0',
                 SymbolSequence([move2key, unlockcorrectdoor, to_room_with_goal, move2goal]))
    task6 = Task('simple-crossing', 'MiniGrid-SimpleCrossingS9N1-v0',
                 SymbolSequence([to_room_with_goal, move2goal]))

    global_transfer_taskset = TaskSet([ task5, task6 ], vocab)
    global_task_setup = TaskSetup([global_taskset, global_transfer_taskset])

def get_task(task_id):
    return global_task_setup.get_task_by_id(task_id)

def get_subtask_id(task, subtask_encoding):
    return task.symbol_vocab.decode(subtask_encoding).id

# NOTE: the following will run every time the script is loaded

### creating the different tasks, and environment combos.
move2goal = Symbol('move2goal', description='TODO')
move2key = Symbol('move2key', description='TODO')
unlockcorrectdoor = Symbol('unlock_correct_door', description='TODO')
to_room_with_goal = Symbol('to_room_with_goal', description='TODO')
subtasks = [move2goal, move2key, unlockcorrectdoor, to_room_with_goal]
vocab = SymbVocabulary(subtasks)


