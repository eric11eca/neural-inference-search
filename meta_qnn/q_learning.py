import os
import numpy as np
import pandas as pd
import state_enumerator as se


class QValues:
    ''' Stores Q_values with helper functions. '''

    def __init__(self):
        self.q = {}

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        values = [
            'start_inference_type',
            'start_premise',
            'start_terminate',
            'end_inference_type',
            'end_premise',
            'end_terminate',
            'utility'
        ]

        for row in zip(*[q_csv[col].values.tolist() for col in values]):
            start_state = se.State(
                inference_type=row[0],
                premise_curr=[1],
                terminate=row[2]
            ).as_tuple()

            end_state = se.State(
                inference_type=row[3],
                premise_curr=[4],
                terminate=row[5]).as_tuple()
            utility = row[6]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [
                    end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

    def save_to_csv(self, q_csv_path):
        start_inference_type = []
        start_premise = []
        start_terminate = []
        end_inference_type = []
        end_premise = []
        end_terminate = []
        utility = []

        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(
                    state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(
                    self.q[start_state_list]['utilities'][to_state_ix])

                start_inference_type.append(start_state.layer_type)
                start_premise.append(start_state.layer_depth)
                start_terminate.append(start_state.terminate)

                end_inference_type.append(to_state.layer_type)
                end_premise.append(to_state.layer_depth)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame(
            {
                'start_inference_type': start_inference_type,
                'start_premise': start_premise,
                'start_terminate': start_terminate,
                'end_inference_type': end_inference_type,
                'end_premise': end_premise,
                'end_terminate': end_terminate,
                'utility': utility
            }
        )
        q_csv.to_csv(q_csv_path, index=False)


class QLeanrer:
    ''' All Q-Learning updates and policy generator
            Args
                state: The starting state for the QLearning Agent
                q_values: A dictionary of q_values --
                                keys: State tuples (State.as_tuple())
                                values: [state list, qvalue list]
                replay_dictionary: A pandas dataframe with columns: 'net' for net strings, and 'accuracy_best_val' for best accuracy
                                            and 'accuracy_last_val' for last accuracy achieved
                output_number : number of output neurons
        '''

    def __init__(
        self, premise, hypothesis, epsilon, state_space_parameters,
        state=None, qstore=None, replaydict=None, WeightInitializer=None,
        device=None,
        replay_dictionary=pd.DataFrame(
            columns=[
                'path', 'epsilon',
                'accuracy_best_val',
                'accuracy_last_val',
                'accuracy_best_test',
                'accuracy_last_test',
                'ix_q_value_update'])):

        self.state_list = []
        self.state_space_parameters = state_space_parameters
        self.enumerator = se.StateEnumerator(state_space_parameters)
        self.state = se.State('start', premise, 0, self.state_list)
        self.qstore = QValues()

        if type(qstore) is not type(None):
            self.qstore.laod_q_values(qstore)
            self.replay_dictionary = pd.read_csv(replaydict, index_col=0)
        else:
            self.replay_dictionary = replay_dictionary

        self.epsilon = epsilon
        self.WeightInitializer = WeightInitializer
        self.device = device

    def update_replay_database(self, new_replay_dic):
        self.replay_dictionary = new_replay_dic

    def gnerate_inference_path(self, epsilon=None):
        if epsilon != None:
            self.epsilon = epsilon
        self._reset_for_new_walk()
        state_list = self._run_agent()
        path_string = self.stringutils.state_list_to_string(state_list)

        if path_string in self.replay_dictionary['path'].values:
            bleu_best = self.replay_dictionary[
                self.replay_dictionary['path'] == path_string]['bleu_best'].values[0]
            sem_sim = self.replay_dictionary[
                self.replay_dictionary['path'] == path_string]['sem_sim'].values[0]
            rouge = self.replay_dictionary[
                self.replay_dictionary['path'] == path_string]['rouge'].values[0]

            self.replay_dictionary = self.replay_dictionary.append(
                pd.DataFrame(
                    [[path_string, bleu_best, sem_sim, rouge, self.epsilon]],
                    columns=['path', 'bleu_best', 'sem_sim', 'rouge', 'epsilon']),
                ignore_index=True)
            self.count += 1
            self.replay_dictionary.to_csv(os.path.join(
                self.save_path, 'replayDict' + str(self.count) + '.csv'))
            self.sample_replay_for_update()
            self.qstore.save_to_csv(os.path.join(
                self.save_path, 'qVal' + str(self.count) + '.csv'))
        else:
            bleu_best = -1.0
            sem_sim = -1.0
            rouge = -1.0

        return (path_string, bleu_best, sem_sim, rouge)

    def save_q(self, q_path):
        self.qstore.save_to_csv(os.path.join(q_path, 'q_values.csv'))

    def _reset_for_new_walk(self):
        '''Reset the state for a new random walk'''

        # Inference Path String
        self.state_list = []

        # Starting State
        # TODO: randomly sample an inference alignment with high confidence
        self.state = se.State('start', 0, 1, 0, 0,
                              self.state_space_parameters.image_size, 0, 0)
        self.bucketed_state = self.enum.bucket_state(self.state)

    def _run_agent(self):
        ''' Have Q-Learning agent sample current policy to generate a network '''
        while self.state.terminate == 0:
            self._transition_q_learning()

        return self.state_list

    def _transition_q_learning(self):
        ''' Updates self.state according to an epsilon-greedy strategy'''
        if self.bucketed_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.bucketed_state, self.qstore.q)

        action_values = self.qstore.q[self.bucketed_state.as_tuple()]
        # epsilon greedy choice
        # TODO: explore new and better exploration stradegy
        if np.random.random() < self.epsilon:
            action = se.State(state_list=action_values['actions'][np.random.randint(
                len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(
                len(action_values['actions'])) if action_values['utilities'][i] == max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = se.State(
                state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        self.bucketed_state = self.enum.bucket_state(self.state)

        self._post_transition_updates()

    def _post_transition_updates(self):
        # State to go in state list
        bucketed_state = self.bucketed_state.copy()
        self.state_list.append(bucketed_state)

    def sample_replay_for_update(self):
        for _ in range(self.state_space_parameters.replay_number):
            path = np.random.choice(self.replay_dictionary['path'])

            # Get reward components for Q-learning
            bleu_best = self.replay_dictionary[
                self.replay_dictionary['path'] == path]['bleu_best'].values[0]
            sem_sim = self.replay_dictionary[
                self.replay_dictionary['path'] == path]['sem_sim'].values[0]
            rouge = self.replay_dictionary[
                self.replay_dictionary['path'] == path]['rouge'].values[0]

            state_list = self.stringutils.convert_model_string_to_states(path)
            state_list = self.stringutils.remove_drop_out_states(state_list)

            # Convert States so they are bucketed
            state_list = [self.enum.bucket_state(
                state) for state in state_list]

            self.update_q_value_sequence(
                state_list, self.build_reward(bleu_best, sem_sim, rouge))

    def build_reward(self, bleu, sim, rouge):
        # TODO: need to define a more sophesticated reward function
        return bleu + sim + rouge

    def update_q_value_sequence(self, states, termination_reward):
        ''' Update all Q-Values for a sequence. '''
        self._update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            self._update_q_value(states[i], states[i+1], 0)

    def _update_q_value(self, start_state, to_state, reward):
        ''' Update a single Q-Value for start_state given the state we transitioned to and the reward. '''
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        max_over_next_states = max(
            self.qstore.q[
                to_state.as_tuple()
            ]['utilities']) if to_state.terminate != 1 else 0

        action_between_states = self.enum.transition_to_action(
            start_state, to_state).as_tuple()

        # Q_Learning update rule
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            self.state_space_parameters.learning_rate * \
            (reward + self.state_space_parameters.discount_factor *
             max_over_next_states - values[actions.index(action_between_states)])

        self.qstore.q[start_state.as_tuple()] = {
            'actions': actions, 'utilities': values}
