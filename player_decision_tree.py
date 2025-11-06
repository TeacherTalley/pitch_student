#!/usr/bin/env python
"""
PlayerDecisionTree Module

This module implements an AI player for the Pitch card game using machine learning
decision trees. The PlayerDecisionTree class subclasses the Player class and replaces
the default play strategy with an intelligent decision-making system based on
position-specific models trained on spreadsheet scenarios.

The module provides a complete framework for AI card-playing including:
- Multiple decision tree models (one for each trick position: 0-3)
- Comprehensive game state feature extraction methods
- Dynamic model selection based on player positioning and out-of-trumps detection
- Scenario loading, expansion, and persistence capabilities
- Command-line interface for model training and analysis

Classes:
    PlayerDecisionTree: AI player using position-specific decision tree models for
        intelligent card selection based on comprehensive game state analysis.

Key Features:
    - Position-aware modeling: Different strategies for leading vs following
    - Adaptive model selection: Handles unexpected positions due to out players
    - Comprehensive feature set: 15 different game state evaluators
    - Action method mapping: Decision tree outputs map to inherited Player methods
    - Scenario management: Load, expand, and persist training scenarios
    - Debug support: Detailed decision tracing and model inspection

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""

import argparse
import os
import pandas as pd

from player import Player
import scenario_decision as sdt


class PlayerDecisionTree(Player):
    """
    AI player implementation using position-specific decision tree models for intelligent
    card play in Pitch games.
    
    This class extends the base Player class with machine learning capabilities, training
    separate decision tree models for each trick position (0-3) to handle the different
    strategic considerations of leading vs following. Models are trained on spreadsheet
    scenarios and use comprehensive game state features to make optimal play decisions.

    Attributes:
        scenario_file (str): Path to the Excel/CSV file containing training scenarios.
        models (list[DecisionTreeClassifier]): Four trained models, one for each trick position.
        scenarios (list[DataFrame]): Original scenario data for each model.
        scenarios_extended (list[DataFrame]): Expanded scenario data with all feature combinations.
        feature_methods (dict[str, callable]): Maps feature names to evaluation methods.
        action_methods (dict[str, callable]): Maps action names to Player class methods.

    Model Architecture:
        - Position 0: First to play (leading) strategies
        - Position 1: Second to play (following lead) strategies  
        - Position 2: Third to play (late following) strategies
        - Position 3: Last to play (closing) strategies
        - Adaptive selection: Uses position 3 model when all followers are out

    Feature Set (15 evaluators):
        Game State: all_trumps, trumps_led, points_in_trick, two_in_trick, three_been_played  
        Hand Analysis: have_2, have_3, have_boss, can_capture_trick
        Player Status: opponent_out, partner_out
        Trick Analysis: opponent_has_trick, partner_has_trick,
                        opponent_played_boss, partner_played_boss

    Strategy Flow:
        1. Extract features from current game state using feature_methods
        2. Select appropriate model based on trick_position and out-player logic
        3. Evaluate model to get action name (e.g., 'play_highest', 'play_off')
        4. Execute corresponding method from action_methods mapping
        5. Return played card with optional debug output

    Error Handling:
        - Validates scenario_file presence in strategies during initialization
        - Checks for missing feature/action method implementations
        - Validates trick position bounds and model availability
        - Raises ValueError for invalid actions or positions

    Integration:
        Inherits all Player capabilities (bidding, hand management) while replacing
        the play_card strategy with intelligent decision tree evaluation. Compatible
        with existing game infrastructure and configuration systems.
    """

    def __init__(self, name=None, position=None, strategies=None, **kwargs):
        """
        Initializes the AI player with decision tree models trained from scenario data.
        
        Calls parent Player constructor then loads and trains four position-specific
        decision tree models from spreadsheet scenarios. Sets up feature and action
        method mappings with validation and warning reports for missing implementations.

        Args:
            name (str, optional): Display name for the player. Defaults to None.
            position (int, optional): Player seat position (0-3). Defaults to None.
            strategies (dict, optional): Strategy configuration containing:
                - scenario_file (str): Required path to Excel/CSV scenario file
                - scenario_sheets (list[int], optional): Sheet indices [0,1,2,3] for models
                Defaults to None.
            **kwargs: Additional player attributes passed to parent constructor.

        Raises:
            ValueError: If 'scenario_file' is not specified in strategies dictionary.

        Initialization Process:
            1. Calls parent Player.__init__ with all parameters
            2. Extracts scenario_file and scenario_sheets from strategies
            3. Loads and trains one model per sheet using
                scenario_decision.load_expand_train_scenarios
            4. Builds unified feature and action name sets from all models
            5. Creates method mapping dictionaries with missing method warnings
            6. Stores all training data and models for later use

        Model Training:
            Each sheet in the scenario file trains one decision tree model using:
            - Original scenarios as training data
            - Extended scenarios with feature combinations
            - Integer conversion for numerical features
            - All models stored in self.models list indexed by position

        Method Mapping:
            Creates dictionaries linking model output names to actual methods:
            - feature_methods: Maps feature names to game state evaluators
            - action_methods: Maps action names to inherited Player play methods
            - Warns about any missing implementations but continues execution
        """
        super().__init__(name=name, position=position, strategies=strategies, **kwargs)
        self.scenario_file = None
        scenario_sheets = [0, 1, 2, 3]
        if strategies and isinstance(strategies, dict):
            self.scenario_file = strategies.get('scenario_file')
            if 'scenario_sheets' in strategies:
                scenario_sheets = strategies['scenario_sheets']
        if not self.scenario_file:
            raise ValueError(
                "A 'scenario_file' must be specified in strategies for PlayerDecisionTree.")

        # Load and train a model for each of the 4 sheets in the scenario file (by name or index)
        self.models = []
        self.scenarios = []
        self.scenarios_extended = []
        feature_set = set()
        action_set = set()
        for sheet in scenario_sheets:
            scenarios, scenarios_extended, model = sdt.load_expand_train_scenarios(
                self.scenario_file, sheet_name=sheet, convert_to_int=True)
            self.scenarios.append(scenarios)
            self.scenarios_extended.append(scenarios_extended)
            self.models.append(model)
            feature_set.update(sdt.get_feature_names(model))
            action_set.update(sdt.get_class_names(model))

        feature_list = sorted(feature_set)
        action_list = sorted(action_set)

        # Build a dictionary mapping feature names to method references
        self.feature_methods = self.map_methods(feature_list)
        # Build a dictionary mapping feature names to method references
        # self.feature_methods = {fname: getattr(self, fname)
        #                         for fname in feature_list if hasattr(self, fname)}
        # Report any features not implemented as methods
        missing = [name for name in feature_list if not hasattr(self, name)]
        if missing:
            print(f"Warning: The following features are not implemented as methods: "
                  f"{missing}")

        # Build a dictionary mapping action names to method references
        self.action_methods = self.map_methods(action_list)
        missing = [name for name in action_list if not hasattr(self, name)]
        if missing:
            print(f"Warning: The following actions are not implemented as methods: "
                  f"{missing}")

    def map_methods(self, method_list):
        """
        Creates a dictionary mapping method names to their corresponding callable methods.
        
        Uses introspection to verify method existence and build a mapping dictionary
        for dynamic method invocation. Only includes methods that actually exist in
        the class to prevent runtime errors.

        Args:
            method_list (list[str]): List of method names to map to callables.

        Returns:
            dict[str, callable]: Dictionary mapping existing method names to their
                corresponding method objects. Missing methods are excluded.
        """
        return {method_name: getattr(self, method_name)
                for method_name in method_list if hasattr(self, method_name)}

    def print_scenarios(self, persist_scenarios=False):
        """
        Displays comprehensive model information and optionally saves extended scenarios.
        
        Outputs detailed information for all trained models including original scenarios,
        extended scenario data with sorting, and decision tree structure visualization.
        Optionally writes all extended scenarios to a multi-sheet Excel file for analysis.

        Args:
            persist_scenarios (bool, optional): If True, creates an Excel file containing
                all extended scenarios with one sheet per model. File is named using
                the base scenario filename with '_extended_scenarios.xlsx' suffix.
                Defaults to False.
        """
        # Prepare to write all models to a single Excel file with each sheet named by 'i'
        excel_writer = None
        if persist_scenarios:
            base_name = os.path.splitext(os.path.basename(self.scenario_file))[0]
            out_filename = f"{base_name}_extended_scenarios.xlsx"
            excel_writer = pd.ExcelWriter(out_filename, engine='openpyxl')

        for i, (scenarios, extended_scenarios, model) in enumerate(zip(self.scenarios,
                                                                       self.scenarios_extended,
                                                                       self.models)):
            print(f"\nModel {i} Scenarios:")
            print(scenarios)

            print(f"\nModel {i} Extended Scenarios:")
            # Sort extended_scenarios by label, then by all feature values before printing
            scenario_df = pd.DataFrame(extended_scenarios)
            # Assume label is the last column
            label_col = scenario_df.columns[-1]
            feature_cols = list(scenario_df.columns[:-1])
            sort_cols = [label_col] + feature_cols
            df_sorted = scenario_df.sort_values(by=sort_cols)
            print(df_sorted)

            if persist_scenarios and excel_writer:
                # Write each sorted dataframe to a separate sheet named by 'i'
                df_sorted.to_excel(excel_writer, sheet_name=str(i), index=False)

            print(f"\nModel {i} Decision Tree:")
            sdt.print_decision_tree(model)

        if persist_scenarios and excel_writer:
            excel_writer.close()
            print(f"All extended scenarios written to {out_filename}")

    def play_scenarios(self, game_state=None):
        """
        Executes intelligent card selection using position-specific decision tree models.
        
        The core AI method that evaluates current game state features, selects the 
        appropriate model based on trick position and player status, then executes
        the predicted action to play a card. Includes sophisticated position adjustment
        logic for handling out-of-trumps players.

        Args:
            game_state (dict, optional): Complete game state dictionary containing:
                - trick_position (int): Current position in trick (0-3)
                - lead (int): Position of trick leader
                - out (list[bool]): Out-of-trumps status for each player
                - plus all other game state needed by feature methods
                Defaults to None (creates empty dict).

        Returns:
            Card: The card selected and played based on decision tree prediction.
            
        Raises:
            ValueError: If trick position is invalid for available models or if the
                predicted action name does not correspond to a valid method.

        Algorithm:
            1. Feature Extraction: Evaluates all features using feature_methods
            2. Position Analysis: Determines expected vs actual trick position
            3. Model Selection: Chooses appropriate model with adjustment logic:
               - Uses expected position if playing out of sequence
               - Uses model 3 (last position) if all followers are out
            4. Decision Making: Evaluates selected model with feature scenario
            5. Action Execution: Calls corresponding method from action_methods
            6. Debug Output: Optional detailed decision tracing

        Debug Features:
            When debug=True in game_state, outputs:
            - Player name and selected model index
            - Predicted action name
            - Current hand contents for decision verification
        """
        if game_state is None:
            game_state = {}

        # Construct scenario dictionary from feature methods
        scenario = {fname: fmethod(game_state)
                    for fname, fmethod in self.feature_methods.items()}

        # Determine which model to use based on the trick position
        trick_position = game_state.get('trick_position', 0)
        lead_position = game_state.get('lead', 0)

        # Check if all players trailing me in the trick are already out
        positions = [(lead_position + offset) % 4 for offset in range(4)]
        expected_position = positions.index(self.position)
        trailing_positions = positions[expected_position + 1:]

        # if playing in unexpected position, implying earlier players are out
        # choose model based upon expected position
        if trick_position < expected_position:
            trick_position = expected_position
            print(f"\t\tWarning: Unexpected position. Use model: {trick_position}")

        trailing_out = all(game_state.get('out', [False]*4)[pos] for pos in trailing_positions)
        if trick_position < 3 and trailing_out:
            trick_position = 3
            # if game_state.get('debug', False):
            #     print(f"\t\tAll following players are out. Use model: {trick_position}")

        if trick_position < 0 or trick_position >= len(self.models):
            raise ValueError(f"Invalid trick position {trick_position} for models.")
        model = self.models[trick_position]
        action = sdt.evaluate_model(model, scenario)

        if game_state.get('debug', False):
            # print(f"{self.name} {model.__class__.__name__} decision: {action}")
            print(f"\t\t{self.name:8} model: {trick_position} decision: {action:25}",
                  f"cards: {self.hand}")
            # scenario_df = pd.DataFrame([scenario])
            # print(scenario_df[0])

        # If the action is a method, call it to play the card
        if action in self.action_methods:
            played_card = self.action_methods[action](game_state=game_state)
        else:
            # throw ValueError if action is not a valid method
            raise ValueError(f"Action '{action}' is not a valid method for PlayerDecisionTree")

        return played_card

    # The following methods are used to evaluate the player's hand and game state.
    # These methods correspond to the features used in the decision tree model.
    # The methods name is listed as the feature name in the decision tree model.
    #
    # If new features are added to the decision tree model,
    # the corresponding methods should be added here.

    def all_trumps(self, game_state=None):
        """
        Returns 1 if all cards in the player's hand are trumps, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain the key 'trumps' (suit).
                Defaults to None.
        Returns:
            int: 1 if all cards are trumps, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        if not self.hand.cards:
            return 0
        return int(all(card.is_trump(game_state.get('trumps', None))
                       for card in self.hand.cards))

    def can_capture_trick(self, game_state=None):
        """
        Determines if the player has any card capable of winning the current trick.
        
        Compares the player's highest available card against the highest card
        already played in the trick to determine if the player can capture it.
        Critical for deciding between aggressive and conservative play strategies.

        Args:
            game_state (dict, optional): Should contain:
                - trick (list[Card]): Cards played by position, None for unplayed
                Defaults to None.

        Returns:
            int: 1 if player can capture the trick, 0 otherwise.

        """
        if game_state is None:
            game_state = {}

        if not self.hand.cards:
            return 0

        cards_played = game_state.get('trick', [])
        # Ignore None values in cards_played for max
        card_ranks = [card.rank for card in cards_played if card is not None]
        highest_played_rank = max(card_ranks, default=0)
        highest_rank_in_hand = max(card.rank for card in self.hand.cards)
        # special case, ignore 2 as highest card since it cannot catch anything
        if highest_rank_in_hand == 2:
            return 0
        return int(highest_rank_in_hand > highest_played_rank)

    def have_2(self, game_state=None):
        """
        Returns 1 if the player has a 2 in their hand, 0 otherwise.

        Args:
            game_state (dict, optional): Not used. Included for interface consistency.
                Defaults to None.
        Returns:
            int: 1 if player has a 2, 0 otherwise.
        """
        if game_state is None:
            game_state = {}
        return int(any(card.rank == 2 for card in self.hand.cards))

    def have_3(self, game_state=None):
        """
        Returns 1 if the player has a 3 in their hand, 0 otherwise.

        Args:
            game_state (dict, optional): Not used. Included for interface consistency.
                Defaults to None.
        Returns:
            int: 1 if player has a 3, 0 otherwise.
        """
        if game_state is None:
            game_state = {}
        return int(any(card.rank == 3 for card in self.hand.cards))

    def have_boss(self, game_state=None):
        """
        Determines if the player holds the highest remaining trump card ("boss").
        
        Compares the highest trump rank in the player's hand against the highest
        rank in the remaining trump cards to identify if the player has the
        current "boss" card that can win any trick.

        Args:
            game_state (dict, optional): Should contain:
                - trumps (str): Current trump suit for is_trump evaluation
                - trumps_remaining (list[int]): Ranks of trumps not yet played
                Defaults to None.

        Returns:
            int: 1 if player has the boss card, 0 otherwise.

        Logic:
            1. Finds highest trump rank in player's hand
            2. Finds highest rank in trumps_remaining list
            3. Returns 1 if they match (player has boss), 0 otherwise
            4. Returns 0 if player has no trump cards

        Strategic Importance:
            The boss card is extremely valuable as it guarantees winning any trick.
            This feature helps the AI decide when to play aggressively vs conservatively
            based on having the ultimate trump advantage.
        """
        if game_state is None:
            game_state = {}

        max_rank = max((card.rank for card in self.hand.cards
                       if card and card.is_trump(game_state.get('trumps', None))), default=0)
        highest_remaining_rank = max((rank for rank in game_state.get('trumps_remaining', [])),
                                     default=0)

        return int(max_rank == highest_remaining_rank)

    def opponent_has_trick(self, game_state=None):
        """
        Determines if an opponent currently has the highest card in the trick.
        
        Analyzes the current trick to see if either opponent (positions +1 or +3
        relative to player) has played the card with the highest rank. Used for
        strategic decisions about whether to try to take the trick or play defensively.

        Args:
            game_state (dict, optional): Should contain:
                - trick (list[Card]): Cards played by position, None for unplayed
                Defaults to None. 

        Returns:
            int: 1 if an opponent has played the highest rank card, 0 otherwise.

        Logic:
            1. Identifies opponent positions: (self.position + 1) % 4 and (self.position + 3) % 4
            2. Finds highest card rank among opponent cards played
            3. Finds overall highest card rank in the trick
            4. Returns 1 if opponent's highest matches overall highest
            5. Returns 0 if no cards played or all cards are "off" (rank 0)
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        if not cards_played:
            return 0

        # Get the highest card of the opponents
        opponent_positions = [(self.position + 1) % 4, (self.position + 3) % 4]
        opponent_cards = [cards_played[pos]
                          for pos in opponent_positions if cards_played[pos]]
        highest_opponent_card = max((card.rank for card in opponent_cards if card is not None),
                                    default=0)

        card_ranks = [card.rank for card in cards_played if card is not None]
        highest_card = max(card_ranks, default=0)

        # if all "off" cards (highest_card == 0), no opponent can have trick
        if highest_card == 0:
            return 0
        return int(highest_opponent_card == highest_card)

    def opponent_out(self, game_state=None):
        """
        Returns 1 if the immediately following opponent is known to be out of trumps, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'out' (dict or list of bools).
                Defaults to None. Assumes opponent is at (self.position + 1) % 4.
        Returns:
            int: 1 if immediately following opponent is out, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        out = game_state.get('out', [])
        if out is None:
            return 0

        opponent_position = (self.position + 1) % 4
        opponent_status = 1 if out[opponent_position] else 0
        return opponent_status

    def opponent_played_boss(self, game_state=None):
        """
        Returns 1 if either opponent (positions +1 or +3 relative to self) has played 
        the highest rank of remaining trumps, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trick' (list of Card objects) 
                and 'trumps_remaining' (list of ranks).  Defaults to None. 
        Returns:
            int: 1 if an opponent has played the highest rank of remaining trumps, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        if not cards_played:
            return 0

        # Get the highest card of the opponents
        opponent_positions = [(self.position + 1) % 4, (self.position + 3) % 4]
        opponent_cards = [cards_played[pos]
                          for pos in opponent_positions if cards_played[pos]]
        highest_opponent_card = max((card.rank for card in opponent_cards if card is not None),
                                    default=0)

        highest_remaining_rank = max((rank for rank in game_state.get('trumps_remaining', [])),
                                     default=0)

        return int(highest_opponent_card == highest_remaining_rank)

    def partner_has_trick(self, game_state=None):
        """
        Returns 1 if the partner has played the highest rank card in the trick so far, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trick' (list of Card objects).
                Defaults to None. Assumes partner is at (self.position + 2) % 4.
        Returns:
            int: 1 if partner has played the highest rank card, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        if not cards_played:
            return 0

        partner_position = (self.position + 2) % 4
        partner_card = cards_played[partner_position]

        if not partner_card:
            return 0

        card_ranks = [card.rank for card in cards_played if card is not None]
        highest_card = max(card_ranks, default=0)
        if highest_card == 0:
            return 0
        return int(partner_card.rank == highest_card)

    def partner_out(self, game_state=None):
        """
        Returns 1 if the partner is known to be out of trumps, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'out' (dict or list of bools).
                Defaults to None. Assumes partner is at (self.position + 2) % 4.
        Returns:
            int: 1 if partner is out, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        out = game_state.get('out', [])
        if out is None:
            return 0

        partner_position = (self.position + 2) % 4
        partner_status = 1 if out[partner_position] else 0
        return partner_status

    def partner_played_boss(self, game_state=None):
        """
        Returns 1 if the partner has played the highest rank of remaining trumps, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trick' (list of Card objects) 
                and 'trumps_remaining' (list of ranks).
                Defaults to None. Assumes partner is at (self.position + 2) % 4.
        Returns:
            int: 1 if partner has played the highest rank of remaining trumps, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        if not cards_played:
            return 0

        partner_position = (self.position + 2) % 4
        partner_card = cards_played[partner_position]

        if not partner_card:
            return 0

        highest_remaining_rank = max((rank for rank in game_state.get('trumps_remaining', [])),
                                     default=0)

        return int(partner_card.rank == highest_remaining_rank)

    def partner_unbeatable(self, game_state=None):
        """
        Returns 1 if the partner played the highest rank of remaining trumps other than 
        those in current player's hand, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trick' (list of Card objects),
                'trumps_remaining' (list of ranks), and 'trumps' (suit).
                Defaults to None. Assumes partner is at (self.position + 2) % 4.
        Returns:
            int: 1 if partner has played the highest remaining trump not in player's hand, 
                 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        if not cards_played:
            return 0

        partner_card = cards_played[(self.position + 2) % 4]
        if not partner_card:
            return 0

        # Check if partner has won the trick so far
        played_ranks = [card.rank for card in cards_played if card is not None]
        if partner_card.rank != max(played_ranks, default=0):
            return 0

        # Get player's trump ranks and find highest available trump not in player's hand
        trump_suit = game_state.get('trumps', None)
        player_trumps = {card.rank for card in self.hand.cards if card.is_trump(trump_suit)}

        trumps_remaining = game_state.get('trumps_remaining', [])
        highest_available = max((rank for rank in trumps_remaining
                                if rank not in player_trumps), default=0)

        return int(partner_card.rank == highest_available)


    def points_in_trick(self, game_state=None):
        """
        Returns the total value of all points played so far in the trick.

        Args:
            game_state (dict, optional): Should contain:
                - trick (list[Card]): Cards played by position, None for unplayed
                Defaults to None.

        Returns:
            int: Total value of all points played so far in the trick.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        return sum(getattr(card, 'points', 0) for card in cards_played if card is not None)

    def three_been_played(self, game_state=None):
        """
        Returns 1 if a card with rank 3 has been played so far in the hand, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trumps_played' (list of ranks).
                Defaults to None.
        Returns:
            int: 1 if a card with rank 3 has been played, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        return int(any(rank == 3 for rank in game_state.get('trumps_played', [])))

    def trumps_led(self, game_state=None):
        """
        Returns 1 if trumps led in the trick, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trumps_led' (bool).
                Defaults to None.
        Returns:
            int: 1 if trumps led, 0 otherwise.
        """
        if game_state and game_state.get('trumps_led'):
            return 1
        return 0

    def two_in_trick(self, game_state=None):
        """
        Returns 1 if a card with rank 2 has been played in the current trick, 0 otherwise.

        Args:
            game_state (dict, optional): Should contain 'trick' (list of Card objects).
                Defaults to None.
        Returns:
            int: 1 if a card with rank 2 has been played in the current trick, 0 otherwise.
        """
        if game_state is None:
            game_state = {}

        cards_played = game_state.get('trick', [])
        return int(any(card.rank == 2 for card in cards_played if card is not None))

if __name__ == "__main__":
    # Command-line interface for PlayerDecisionTree model training and analysis.
    #
    # Provides a standalone tool for loading scenario files, training decision tree
    # models, and generating detailed analysis outputs including model visualization
    # and extended scenario data exports.
    #
    # Usage:
    #     python player_decision_tree.py scenarios.xlsx
    #     python player_decision_tree.py scenarios.xlsx --sheets 0 1 2 3
    #
    # Arguments:
    #     scenario_file: Path to Excel/CSV file containing training scenarios
    #     --sheets: Optional list of sheet indices to process (default: [0,1,2,3])
    #
    # Outputs:
    #     - Model information printed to console
    #     - Decision tree visualizations for each model
    #     - Extended scenarios exported to Excel file with '_extended_scenarios.xlsx' suffix

    parser = argparse.ArgumentParser(description="PlayerDecisionTree scenario model builder")
    parser.add_argument("scenario_file", type=str, help="Path to scenario spreadsheet file")
    parser.add_argument("--sheets", nargs="*", type=int, default=[0, 1, 2, 3],
                        help="List of sheet indices to build models for")
    args = parser.parse_args()

    # Instantiate PlayerDecisionTree with the provided scenario file and sheets
    strategies_arg = {'scenario_file': args.scenario_file, 'scenario_sheets': args.sheets}
    player = PlayerDecisionTree(name="TestPlayer", position=0, strategies=strategies_arg)

    player.print_scenarios(persist_scenarios=True)
