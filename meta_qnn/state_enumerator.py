import math
import numpy as np
from operator import itemgetter


class Action:
    def __init__(self, inference_type=None):
        if inference_type is None:
            self.inference_type = "exact_align"
        else:
            self.inference_type = inference_type

        self.curr_alignments = {}

    def generate_alignments(self):
        pass

    def alignment_classification(self):
        pass

    def generate_next_state(self, premise_curr):
        transition = self.curr_alignments[self.inference_type]
        span_orig = transition[0]
        span_next = transition[1]
        new_state = State(
            inference_type=self.inference_type,
            premise_curr=premise_curr.replace(span_orig, span_next),
            terminate=False)
        return new_state


class State:
    '''
    inference_type: String -- paraphrase, commonsense, monotonicity, ...
    premise_prev: String -- the previous premise
    premise_curr: String -- the current premise
    '''

    def __init__(self, inference_type=None, premise_curr="", terminate=None, state_list=None):
        if not state_list:
            self.inference_type = inference_type
            self.premise_curr = premise_curr
            self.terminate = terminate
        else:
            self.inference_type = state_list[0]
            self.premise_curr = state_list[1]
            self.terminate = state_list[2]

    def as_tuple(self):
        return (self.inference_type,
                self.premise_curr,
                self.terminate)

    def as_list(self):
        return list(self.as_tuple())

    def copy(self):
        return State(
            self.inference_type,
            self.premise_curr,
            self.terminate
        )


class StateEnumerator:
    '''
    Enumerating States (defining their possible transitions)
    '''

    def __init__(self, state_space_parameters):
        # Limits
        self.output_states = state_space_parameters.output_states

    def enumerate_state(self, state, q_values):
        '''
        Defines all state transitions, populates q_values where actions are valid
        Updates: q_values and returns q_values
        '''
        actions = []

        # TODO: API interface for the fine-tuned Language Model vertex probing (Bert)
        # TODO: Define action space here, i.e. rerun the probing framework (Eric)

        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {'actions': [self.bucket_state_tuple(to_state.as_tuple()) for to_state in actions],
                                      'utilities': [self.ssp.init_utility for i in range(len(actions))]}
        return q_values

    def transition_to_action(self, start_state, to_state):
        action = to_state.copy()
        if to_state.layer_type not in ['fc', 'gap']:
            action.image_size = start_state.image_size
        return action

    def state_action_transition(self, start_state, action):
        ''' start_state: Should be the actual start_state, not a bucketed state
            action: valid action

            returns: next state, not bucketed
        '''
        to_state = action.generate_next_state(start_state.premise_curr)
        return to_state

    def bucket_state_tuple(self, state):
        bucketed_state = State(state_list=state).copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(
            bucketed_state.image_size)
        return bucketed_state.as_tuple()

    def bucket_state(self, state):
        bucketed_state = state.copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(
            bucketed_state.image_size)
        return bucketed_state

    def _possible_conv_sizes(self, image_size):
        return [conv for conv in self.ssp.possible_conv_sizes if conv < image_size]

    def _possible_pool_sizes(self, image_size):
        return [pool for pool in self.ssp.possible_pool_sizes if pool < image_size]

    def _possible_pool_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_pool_strides if stride <= filter_size]

    def _possible_fc_size(self, state):
        '''Return a list of possible FC sizes given the current state'''
        if state.layer_type == 'fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes
