import numpy as np

class ProceduralGraph(object):
    """
    This expresses rooms as nodes of a graph, connected by edges if they are connected by a hallway  or door of some kind. Edges can be either hallway, unlocked door, or locked.
    Rooms may also contain keys and goals.
    """

    def __init__(self, subtask_sequence, cluster_max_size=2, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.cluster_max_size = cluster_max_size

        self.subtask_sequence = subtask_sequence
        self.rooms = []
        self.key_counter = 1 # key numbering starts at 1
        self.room_counter = 0

        self.accessible = []
        self.connections = {} # keys are going to be a tuple of room indices.
        self.possessions = {} # keys are going to be room indices
        #  used keys list
        self.used_keys = set([])

        # generating the first accessible rooms and their connections
        inaccessible = self._make_room_cluster()
        self.accessible.extend(inaccessible)
        self.starting_point = np.random.choice(self.accessible)
        # walking through the subtask sequence to create the graph
        self.parse_subtasks()

        self.connections_table = self.convert_connection()

    def parse_subtasks(self):
        for subtask in self.subtask_sequence:
            subtask_id = subtask.id
            if subtask_id == 'to_room_with_goal':
                print('toroomwithgoal encountered. Placing goal.')
                # then put the goal in one of the accessible rooms
                room_with_goal = np.random.choice(self.accessible)
                self.possessions[room_with_goal]['goal'] = True
            elif subtask_id == 'move2key':
                print('move2key encountered: Placing key')
                # then put a key (of the current key_counter) in one of the accessible rooms
                room_with_key = np.random.choice(self.accessible)
                self.possessions[room_with_key]['key'].append(self.key_counter)
                # increment self.key_counter
                self.key_counter += 1
            elif subtask_id == 'unlock_correct_door':
                print('unlock encountered: Adding new room cluster')
                # create a new room, put into a new room cluster (and generate )
                inaccessible = self._make_room_cluster()
                # choose an index from any of the unused keys
                available_keys = list(set(range(1, self.key_counter)) - self.used_keys)
                if len(available_keys) == 0:
                    raise ValueError("sequence seems to suggest that there's a missing key.")
                key_idx = np.random.choice(available_keys)
                # choose some room within the current accessible clusters
                accessible_room = np.random.choice(self.accessible)
                # choose some room within the new clusters
                inaccessible_room = np.random.choice(inaccessible)
                # create a lock door between them corresponding to a certain key
                self.connections[(accessible_room, inaccessible_room)] = key_idx
                self.connections[(inaccessible_room, accessible_room)] = key_idx
                # append the current cluster into the accessible list
                self.accessible.extend(inaccessible)
                # update the set of used keys.
                self.used_keys.add(key_idx)

    def _make_room_cluster(self):
        cluster_size = np.random.randint(1, self.cluster_max_size+1)
        inaccessible = []
        for _ in range(cluster_size):
            self.rooms.append(self.room_counter)
            self.possessions[self.room_counter] = {'goal': False, 'key': []}
            inaccessible.append(self.room_counter)
            if len(inaccessible) > 1:
                connection_type = np.random.choice(['unlocked', 'hallway'])
                self.connections[(self.room_counter-1, self.room_counter)] = connection_type
                self.connections[(self.room_counter, self.room_counter-1)] = connection_type
            self.room_counter += 1
        return inaccessible

    def convert_connection(self):
        table = np.zeros((self.room_counter, self.room_counter)).tolist()
        for i in range(len(table)):
            for j in range(len(table[i])):
                if (i, j) in self.connections:
                    table[i][j] = self.connections[(i, j)]
        return table

    def get_room_pos(self):
        """
        The rooms need to have different positions, but this needs to be derived from the graph of rooms
        Outputs:
            grid_dimensions
            num_cols, num_rows
            position to idx mapping
            psoition to possessions
        """

        # left, right, up, down
        deltas = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        # room 0,0 is the starting point.
        to_add = [(self.starting_point, None)]
        pos = np.array([0, 0])
        pos2idx = {}
        idx2pos = {}

        while len(to_add):
            this_room, parent = to_add.pop(0)
            # assign a placement that doens't collide with previously assigned positions.
            if parent is None: # this means it's a root ndoe
                pos2idx[tuple(pos.tolist())] = this_room
                idx2pos[this_room] = tuple(pos.tolist())
            else:
                assert parent in idx2pos
                parent_pos = np.array(idx2pos[parent])
                pos_choices = parent_pos + deltas
                pos_valid_choices = [idx for idx, el in enumerate(pos_choices) if tuple(el.tolist()) not in pos2idx]
                assert len(pos_valid_choices), "Ran out so of choices for room positions!"
                pos_idx = np.random.choice(pos_valid_choices)
                pos2idx[tuple(pos_choices[pos_idx].tolist())] = this_room
                idx2pos[this_room] = tuple(pos_choices[pos_idx].tolist())
            # breadth first search,putting rooms in different places.
            connected_neighbors = [(idx, this_room) for idx, val in enumerate(self.connections_table[this_room]) if val != 0 and idx not in idx2pos]
            to_add.extend(connected_neighbors)

        return idx2pos, pos2idx

    def get_rowcol(self):
        idx2pos, pos2idx = self.get_room_pos()
        # get the min and the max along each row and column.
        positions = np.array(list(pos2idx.keys()))
        pos_min = np.min(positions, axis=0)
        pos_max = np.max(positions, axis=0)
        # first element tells us the index along rows, second element tells us the index along columns
        num_rows, num_cols = pos_max - pos_min + 1
        rowcol = np.ones((num_rows, num_cols)) * -1
        # Filling out the row col matrix
        self.room2pos = {}
        for i in range(num_rows):
            for j in range(num_cols):
                offset_pos = (i + pos_min[0], j + pos_min[1])
                if offset_pos in pos2idx:
                    rowcol[i][j] = pos2idx[offset_pos]
                    self.room2pos[pos2idx[offset_pos]] = (i,j)
        # insertion of doors and hallways
        # door creation list
        # locked doors
        locked = []
        hallways = []
        unlocked = []
        for room1, room2 in self.connections:
            # we know that the locked will be added towards the right and down.
            row1, col1 = self.room2pos[room1]
            row2, col2 = self.room2pos[room2]
            if (row1 == row2 and col1+1 == col2):
                if isinstance(self.connections[(room1, room2)], np.int64) and self.connections[(room1, room2)] >= 1:
                    locked.append((room1, self.connections[(room1, room2)], 'right'))
                elif self.connections[(room1, room2)] == 'hallway':
                    hallways.append((room1, 'right'))
                elif self.connections[(room1, room2)] == 'unlocked':
                    unlocked.append((room1, 'right'))
            elif (row1+1 == row2 and col1 == col2):
                if isinstance(self.connections[(room1, room2)], np.int64) and self.connections[(room1, room2)] >= 1:
                    locked.append((room1, self.connections[(room1, room2)], 'down'))
                elif self.connections[(room1, room2)] == 'hallway':
                    hallways.append((room1, 'down'))
                elif self.connections[(room1, room2)] == 'unlocked':
                    unlocked.append((room1, 'down'))

            # for every single door, figure out which two rooms it belongs to
        return rowcol, locked, unlocked, hallways
