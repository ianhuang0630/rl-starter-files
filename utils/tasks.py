"""
Implementation of a task set
"""

class TaskSetup(object):
    def __init__(self, task_sets):
        assert len(task_sets) == 2, 'other lengths not supported at this moment'

        self.unique_tasks = set([])
        for task_set in task_sets:
            for task in task_set:
                self.unique_tasks.add(task.id)
        taskid2idx = {taskid: idx  for idx, taskid in enumerate(list(self.unique_tasks))}

        for task_set in task_sets:
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

    @property
    def transfer_task_set(self):
        return self.transfer_set

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

class Task(object):
    def __init__(self, name, environment, symbol_sequence):
        self.name = name
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
        return "{}: {}  |  {}".format(self.name, self.env_str, str(self.symbol_seq))

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
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}

    def encode(self, symbol):
        assert symbol in self.symbol2idx, "Invalid symbol"
        rep = [0] * self.size
        rep[self.symbol2idx[symbol]] = 1
        return rep

    def __getitem__(self, item):
        return self.symbols[item]

    def __str__(self):
        return "\n".join([str(symb) for symb in self.symbols])

    @property
    def size(self):
        return len(self.symbols)

# NOTE: the following will run every time the script is loaded

### creating the different tasks, and environment combos.
move2goal = Symbol('move2goal', description='TODO')
pickupgoal = Symbol('pickupgoal', description='TODO')
move2key = Symbol('move2key', description='TODO')
unlockcorrectdoor = Symbol('unlock_correct_door', description='TODO')
to_room_with_goal = Symbol('to_room_with_goal', description='TODO')

# Task 1: 4-room senvironment
# TODO: get the environment
task1 = Task('to-room-with-goal', 'MiniGrid-FourRooms-ToRoomWithGoal-v0',
             SymbolSequence([to_room_with_goal]))
task2 = Task('to-goal', 'MiniGrid-FourRooms-Get2Goal-v0',
             SymbolSequence([move2goal]))
task3 = Task('4-room', 'MiniGrid-FourRooms-v0',
             SymbolSequence([to_room_with_goal, move2goal]))
task4 = Task('unlock', 'MiniGrid-Unlock-v0',
             SymbolSequence([move2key, unlockcorrectdoor, to_room_with_goal, move2goal]))

vocab = SymbVocabulary([move2goal, pickupgoal, move2key, unlockcorrectdoor, to_room_with_goal])
global_taskset = TaskSet([task1, task2, task3, task4], vocab)

task5 = Task('door-key', 'MiniGrid-DoorKey-5x5-v0',
             SymbolSequence([move2key, unlockcorrectdoor, to_room_with_goal, move2goal]))
task6 = Task('simple-crossing', 'MiniGrid-SimpleCrossingS9N1-v0',
             SymbolSequence([to_room_with_goal, move2goal]))
global_transfer_taskset = TaskSet([ task5, task6 ], vocab)

global_task_setup = TaskSetup([global_taskset, global_transfer_taskset])

