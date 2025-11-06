#!/usr/bin/env python
"""
Player Module

This module defines the Player class for the Pitch card game, representing individual
players with comprehensive bidding and playing strategies. The module supports both
built-in strategies and external strategy modules, configurable player personalities
through aggressiveness and restraint parameters, and Excel-based bid strength
customization for advanced gameplay analysis.

Classes:
    Player: A comprehensive player class managing hand state, bidding logic, and
        multiple playing strategies with configurable behavior parameters.

Functions:
    get_args(): Parses command line arguments for player configuration and game modes.
    load_config(config_file): Loads YAML configuration files for player setup.
    configure_players(config, game_state): Creates player objects from configuration.
    main(): Demonstrates player functionality with bidding and hand management.

Module Constants:
    default_bid_strength (MappingProxyType): Immutable mapping of card ranks to their
        bidding strength values, with special handling for Pitch-specific cards like
        jacks and jokers.

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""

from types import MappingProxyType
from pprint import pprint
import importlib
import argparse
import random
import yaml
import pandas as pd

# from card import Card
from hand import Hand
from deck import Deck

# make the default strength dictionary immutable
default_bid_strength = MappingProxyType({17: 2.5,
                                         16: 1.5,
                                         15: 1.25,
                                         14: 0.75,  # Main Jack
                                         13: 0.75,  # Off Jack
                                         12: 0.5,   # Big Joker
                                         11: 0.5,   # Little Joker
                                         10: 0.5,
                                         9: 0.25,
                                         8: 0.25,
                                         7: 0.1,
                                         6: 0.1,
                                         5: 0.1,
                                         4: 0.1,
                                         3: 1.5,
                                         2: 1,
                                         1: 0,
                                         0: 0
                                         })


class Player:
    """
    The Player class manages all aspects of gameplay including hand management,
    bidding with suit evaluation, trump selection, and multiple playing strategies.
    Supports both internal methods and external strategy modules, with configurable
    aggressiveness and restraint parameters affecting bidding behavior.

    Attributes:
        name (str): The display name of the player.
        position (int): The player's seat position (0-3) in a 4-player game.
        hand (Hand): The Hand object containing the player's current cards.
        bid_value (int): The numerical value of the player's current bid.
        bidding_suit (str): The trump suit the player would choose if they win the bid.
        strategies (dict): Configuration dictionary containing:
            - bid_strategy (callable): Function for making bids
            - play_strategy (callable): Function for selecting cards to play
            - aggressiveness (float): Added to computed bid values (default 0.0)
            - restraint (float): Subtracted from bids when partner is leading (default 0.0)
            - bid_strength (MappingProxyType): Mapping of card ranks to strength values
            - *_external (bool): Flags indicating if strategies are external modules

    Strategy Parameters:
        aggressiveness: Positive values increase bidding likelihood by adding to
            computed bid strength. Represents confidence in partner support and
            deck drawing luck.
        restraint: Positive values decrease bidding when partner leads, preventing
            overbidding. If computed bid minus restraint ≤ partner's bid, player passes.
    """


    def __init__(self, name=None, position=None, strategies=None, **kwargs):
        """
        Initializes a new Player instance with configurable strategies and attributes.
        
        Sets up default strategies and personality parameters, then processes any
        provided strategy overrides. Handles both internal class methods and external
        module imports for bid and play strategies. Supports Excel-based bid strength
        customization through the 'bid_strength_file' strategy parameter.

        Args:
            name (str, optional): Display name for the player. Defaults to None.
            position (int, optional): Seat position (0-3) in 4-player game. Defaults to None.
            strategies (dict, optional): Strategy configuration dictionary that may contain:
                - bid_strategy (str): Method name or 'module.method' for bidding
                - play_strategy (str): Method name or 'module.method' for card play
                - aggressiveness (float): Bidding confidence modifier
                - restraint (float): Partner overbidding prevention modifier
                - bid_strength_file (str): Excel file path for custom bid strengths
                - bid_strength_sheet (int/str): Sheet index/name in Excel file
                Defaults to None.
            **kwargs: Additional attributes to set on the player instance.

        Strategy Processing:
            - External strategies (containing '.') are imported as 'module.method'
            - Internal strategies reference methods within this class or subclasses
            - Excel bid strength files override default_bid_strength mappings
            - Missing strategies fall back to default_bid_strategy and default_play_strategy
        """
        self.name = name
        self.position = position

        # player's hand state
        self.hand = Hand()
        self.bid_value = 0
        self.bidding_suit = None

        # set default player strategies
        self.strategies = {}
        self.strategies = {'bid_strategy': self.default_bid_strategy,
                           'play_strategy': self.default_play_strategy,
                           'aggressiveness': 0.0,
                           'restraint': 0.0, 
                           'bid_strength': default_bid_strength
                           }

        # check for bid_strength_file and load if specified
        if strategies and 'bid_strength_file' in strategies:
            bid_strength_dict = self.load_bid_strength(strategies['bid_strength_file'],
                                                    strategies.get('bid_strength_sheet', 0))
            if bid_strength_dict:
                self.strategies['bid_strength'] = MappingProxyType(bid_strength_dict)

        # set passed strategies
        # print(f"Setting strategies for {self.name}: {strategies}")
        if strategies:
            for key, value in strategies.items():
                # print(f"Setting strategy {key} to {value} in {self.name}")
                if key in self.strategies:
                    if key in ('bid_strategy', 'play_strategy'):
                        # if the strategy contains a '.', assume method not in class
                        # try to import the module and get the method
                        if '.' in value:
                            # print(f"Setting {key} to {value} as external in {self.name}")
                            module_name, method_name = value.rsplit('.', 1)
                            module = importlib.import_module(module_name)
                            self.strategies[key] = getattr(module, method_name)
                            self.strategies[key + '_external'] = True
                        else:
                            # assume the strategy is a method in this class or subclass
                            # print(f"Setting {key} to {value} as subclass in {self.name}")
                            self.strategies[key] = getattr(self, value)
                            self.strategies[key + '_external'] = False
                    else:
                        self.strategies[key] = value

        # set other passed attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_bid_strength(self, bid_strength_file, bid_strength_sheet=0):
        """
        Loads bid strength values from an Excel file.

        This method reads an Excel file containing bid strength data and creates
        a dictionary mapping card ranks to their corresponding strength values.
        The Excel file should contain columns named 'rank' and 'strength'.

        Args:
            bid_strength_file (str): Path to the Excel file containing bid strength data.
            bid_strength_sheet (int, optional): Sheet index or name to read from. 
                Defaults to 0 (first sheet).

        Returns:
            MappingProxyType or None: An immutable dictionary mapping card ranks (int) 
                to strength values (float), or None if the file could not be loaded.

        Raises:
            Prints warning messages for FileNotFoundError, EmptyDataError, KeyError, 
            or ValueError exceptions and returns None.

        Note:
            The Excel file must contain columns named 'rank' and 'strength'.
            If loading fails, a warning message is printed and None is returned,
            allowing the caller to fall back to default bid strength values.
        """
        try:
            # Get all sheet names in the Excel file
            # excel_file = pd.ExcelFile(bid_strength_file)
            # print(f"Sheets found in {bid_strength_file}: {excel_file.sheet_names}")

            # Read the Excel file
            scenario_df = pd.read_excel(bid_strength_file, sheet_name=bid_strength_sheet)

            # Create dictionary mapping rank to strength
            bid_strength_dict = {}
            for _, row in scenario_df.iterrows():
                rank = row['rank']
                strength = row['strength']
                bid_strength_dict[rank] = strength

            # Update the bid_strength in strategies
            return MappingProxyType(bid_strength_dict)

        except (FileNotFoundError,
                pd.errors.EmptyDataError,
                KeyError,
                ValueError) as file_error:
            print("Warning: Could not load bid strength file ",
                    f"{bid_strength_file}: {file_error}")
            print("Using default bid strength values")
        return None


    def __str__(self):
        """
        Returns the string representation of the player.

        Returns:
            str: The name of the player.
        """
        return self.name

    def __repr__(self):
        """
        Returns the string representation of the player.

        Returns:
            str: The string representation of the player.
        """
        return self.name

    def state(self):
        """
        Returns a dictionary representation of the player's attributes.

        Returns:
            dict: A dictionary containing the player's attributes.
        """
        return vars(self)

    def is_partner(self, player_position=None):
        """
        Checks if the given player position is a partner (2 seats away in a 4-player game).
        Note: Returns False if player_position is None, 0, or any falsy value.

        Args:
            player_position (int, optional): The position of the player to check. Defaults to None.

        Returns:
            bool: True if the given player position is a partner, False otherwise.
        """
        if not player_position:
            return False
        return self.position == (player_position + 2) % 4

    def default_bid_strategy(self, current_bid=0, current_bidder=None, game_state=None):
        """
        Evaluates all four suits and makes an intelligent bid considering aggressiveness,
        restraint, and current bidding situation.
        
        The strategy performs comprehensive suit evaluation by temporarily setting each 
        suit as trump and calculating total strength using the bid_strength mapping.
        Applies personality modifiers (aggressiveness/restraint) and enforces Pitch
        bidding rules including minimum bids, overbidding prevention, and partner restraint.

        Args:
            current_bid (int, optional): The current highest bid value. Defaults to 0.
            current_bidder (int, optional): The position of the current highest bidder. 
                Used for partner detection and restraint application. Defaults to None.
            game_state (dict, optional): Game state dictionary that may contain:
                - debug (bool): Enables detailed bidding output. Defaults to None.

        Returns:
            int: The final bid value (0 = pass, 5-7 = actual bid). Also sets:
                - self.bid_value: The computed bid amount
                - self.bidding_suit: The optimal trump suit for this bid

        Bidding Logic:
            1. Evaluates each suit by temporarily setting trump and summing card strengths
            2. Selects suit with highest total strength value
            3. Applies aggressiveness modifier (added to strength)
            4. Applies restraint modifier if partner is current high bidder
            5. Enforces minimum bid of 5 (passes if below)
            6. Caps maximum bid at 7
            7. Passes if unable to exceed current_bid
            8. Bids exactly one over current_bid when possible
            9. Prevents bidding 7 when no opposition exists

        Debug Output:
            When debug=True, prints detailed suit evaluations, strength calculations,
            and final bid decisions with aggressiveness/restraint values.
        """
        if game_state is None:
            game_state = {}

        debug = game_state.get('debug', False)
        aggressiveness = self.strategies.get('aggressiveness', 0.0)
        restraint = self.strategies.get('restraint', 0.0)

        if debug:
            print(f"{self.name:<10} ",
                  f"(aggressiveness: {aggressiveness:<.1f}, restraint: {restraint:<.1f}) ",
                  f"bidding with {self.hand}")

        bid_strength = self.strategies.get('bid_strength', default_bid_strength)

        self.bid_value = None
        self.bidding_suit = None
        for suit in ['Spades', 'Hearts', 'Clubs', 'Diamonds']:
            # tentatively set trumps to evaluate bid
            self.hand.set_trump(suit)
            trump_ranks = [bid_strength[card.rank] for card in self.hand]
            suit_bid = round(sum(trump_ranks), 2)

            if debug:
                print(f"\t{suit:8} for {suit_bid:2.2f} {trump_ranks}")

            if not self.bid_value or suit_bid > self.bid_value:
                self.bid_value = suit_bid
                self.bidding_suit = suit

        # ignore our tentative setting of trumps
        self.hand.set_trump(None)

        # add aggressiveness to bid and truncate to integer
        self.bid_value = self.bid_value + aggressiveness

        if self.is_partner(current_bidder):
            # if partner currently has bid, subtract restraint from bid value
            self.bid_value -= restraint
            self.bid_value = max(0, self.bid_value)  # ensure bid_value is not negative

        if debug:
            print(f"\tComputed bid: {self.bid_value:<.2f} in {self.bidding_suit}")

        self.bid_value = int(self.bid_value)

        if self.bid_value < 5:  # can't meet minimum bid, so pass
            self.bid_value = 0
        # Do not limit maximum bid here; let overbidding logic handle it
        #elif self.bid_value > 7:  # let's not go overboard
        #    self.bid_value = 7

        # if current bid is higher than our bid, pass
        if self.bid_value <= current_bid:
            self.bid_value = 0
        elif self.bid_value > current_bid > 0:  # bid just over current bid
            self.bid_value = current_bid + 1
        elif self.bid_value >= 7:  # don't bid 7 since there is no opponent bid
            self.bid_value = 6

        return self.bid_value

    def bid(self, current_bid=0, current_bidder=None, game_state=None):
        """
        Delegates to the configured bid strategy function to make a bid.
        
        This method serves as the public interface for bidding, allowing the player
        to use different bidding strategies (default, external modules, or subclass
        methods) without changing the calling code.

        Args:
            current_bid (int, optional): The current highest bid value. Defaults to 0.
            current_bidder (int, optional): The position of the current highest bidder. 
                Defaults to None.
            game_state (dict, optional): The current state of the game containing
                contextual information for strategy decisions. Defaults to None.

        Returns:
            int: The bid made by the configured strategy (0 = pass, 5-7 = actual bid).
        """
        return self.strategies['bid_strategy'](current_bid=current_bid,
                                               current_bidder=current_bidder,
                                               game_state=game_state)

    def choose_trumps(self):
        """
        Returns the trump suit determined during the bidding process.
        
        Returns:
            str: The trump suit chosen by the player ('Spades', 'Hearts', 'Clubs', 
                or 'Diamonds'), or None if no bidding has occurred.
        """
        return self.bidding_suit

    def default_play_strategy(self, game_state=None):
        """
        Simple default playing strategy that plays the first card in hand.
        
        Args:
            game_state (dict, optional): The current state of the game. Not used by
                this simple strategy but maintained for interface consistency. 
                Defaults to None.

        Returns:
            Card: The first card from the player's hand.
        """
        if game_state is None:
            game_state = {}
        played_card = self.hand[0]
        return played_card

    def play_card(self, game_state=None):
        """
        Executes the configured play strategy with error handling and fallback logic.
        
        This method serves as the primary interface for card playing, delegating to
        the configured play_strategy while handling both internal and external strategy
        implementations. Includes robust error handling that falls back to random
        play if the strategy fails to return a valid card.

        Args:
            game_state (dict, optional): The current state of the game containing
                contextual information for strategy decisions. Defaults to None.

        Returns:
            Card: The card selected by the strategy (or random fallback if strategy fails).

        Strategy Handling:
            - External strategies: Called with (self, game_state=game_state)
            - Internal strategies: Called with (game_state=game_state) only
            - Strategy type determined by play_strategy_external flag

        Error Recovery:
            If the strategy returns None or an invalid card, prints an error message
            identifying the failed strategy and automatically falls back to play_random()
            to ensure gameplay continues.

        Side Effects:
            Always removes the selected card from the player's hand before returning it.
        """
        if self.strategies.get('play_strategy_external', False):
            # implementation is in this class or is a subclass of this class
            played_card = self.strategies['play_strategy'](self, game_state=game_state)
        else:
            # implementation is external module without subclassing
            played_card = self.strategies['play_strategy'](game_state=game_state)

        if not played_card:
            print(f"ERROR:{self.name} strategy {self.strategies['play_strategy'].__name__} ",
                  f"did not select a card from {self.hand}")
            print("Selecting random card instead.")
            played_card = self.play_random(game_state=game_state)

        # remove the played card from the player's hand
        self.hand.remove_cards(played_card)

        return played_card

    def play_random(self, game_state=None):
        """
        Plays a random card from the player's hand. If trumps are led and the
        player has trump cards, plays a random trump card instead.

        Args:
            game_state (dict, optional): The current state of the game. Should
                contain the keys 'trumps' (str) for the trump suit and 
                'trumps_led' (bool). Defaults to None.

        Returns:
            Card: A randomly selected card from the player's hand.
        """
        if game_state is None:
            game_state = {}

        trump_cards = [card for card in self.hand.cards
                       if card.is_trump(suit=game_state.get('trumps', None))]
        if game_state.get('trumps_led', False) and trump_cards:
            played_card = random.choice(trump_cards)
        else:
            played_card = random.choice(self.hand.cards)
        return played_card

    def play_highest(self, game_state=None):
        """
        Plays the highest ranking card from the player's hand.

        Args:
            game_state (dict, optional): The current state of the game. Defaults to None.

        Returns:
            Card: The highest ranking card played by the player.
        """
        if game_state is None:
            game_state = {}
        return max(self.hand.cards, key=lambda card: card.rank)

    def play_lowest(self, game_state=None):
        """
        Plays the lowest ranking card from the player's hand.
        If trumps are led, plays the lowest trump card instead.

        Args:
            game_state (dict, optional): The current state of the game. Should
                contain the keys 'trumps_led' (bool) and 'trumps' (str). Defaults to None.

        Returns:
            Card: The lowest ranking card played by the player.
        """
        if game_state is None:
            game_state = {}

        # if trumps led, play the lowest trump card
        if game_state.get('trumps_led', False):
            trump_cards = [card for card in self.hand.cards
                           if card.is_trump(suit=game_state.get('trumps', None))]
            if trump_cards:
                return min(trump_cards, key=lambda card: card.rank)
        return min(self.hand.cards, key=lambda card: card.rank)


    def play_highest_no_point(self, game_state=None):
        """
        Plays the highest ranked trump card from the player's hand that has zero
        point value. If no zero-point trump cards exist, plays the highest 
        ranking card from the entire hand.

        Args:
            game_state (dict, optional): The current state of the game. Should
                contain the key 'trumps' (str) for the trump suit. Defaults to None.

        Returns:
            Card: The card played by the player.
        """
        if game_state is None:
            game_state = {}

        trump_cards = [card for card in self.hand.cards
                       if card.is_trump(suit=game_state.get('trumps', None))]
        zero_point_cards = [card for card in trump_cards if card.points == 0]
        if zero_point_cards:
            return max(zero_point_cards, key=lambda card: card.rank)
        return max(self.hand.cards, key=lambda card: card.rank)

    def play_highest_points(self, game_state=None):
        """
        Plays the card with the highest point value, prioritizing "catchable" cards.
        
        Args:
            game_state (dict, optional): The current state of the game. Should
                contain the key 'trumps' (str) for trump suit identification and
                'trumps_led' (bool) for trump-following logic. Defaults to None.

        Returns:
            Card: The selected card based on the following priority:
                1. Highest "catchable" point cards (excluding rank 2 and 17)
                2. The 2 of trumps if it's the only point card available
                3. Lowest trump card if trumps were led
                4. Lowest ranked card (likely "off" card) otherwise

        Strategy Logic:
            - Finds all point cards and selects those with maximum point value
            - Excludes Ace (rank 17) as it's uncatchable and valuable for winning
            - Excludes 2 (rank 2) unless it's the only point card available
            - Among eligible cards, chooses the one with lowest rank for safety
            - Falls back to appropriate trump or off-card play if no point cards
        """
        if game_state is None:
            game_state = {}

        # Check for point cards
        point_cards = [card for card in self.hand.cards if card.points > 0]
        if point_cards:
            max_points = max((card.points for card in point_cards))
            highest_point_cards = [card for card in point_cards if card.points == max_points]

            # exclude the Ace from consideration for points
            highest_point_cards = [card for card in highest_point_cards if card.rank != 17]

            # Exclude 2s unless it's the only point card
            catchable_cards = [card for card in highest_point_cards if card.rank != 2]
            if catchable_cards:
                chosen_card = min(catchable_cards, key=lambda card: card.rank)
                return chosen_card

            # play the 2 if it's the only point card
            if highest_point_cards:
                chosen_card = min(highest_point_cards, key=lambda card: card.rank)
                return chosen_card

        # if trumps led, play lowest trump card; otherwise don't waste a trump
        trumps_led = game_state.get('trumps_led', False)
        if trumps_led:
            trump_cards = [card for card in self.hand.cards
                           if card.is_trump(suit=game_state.get('trumps', None))]
            if trump_cards:
                return min(trump_cards, key=lambda card: (card.points, card.rank))

        # If no point cards, play the lowest card (likely an "off" card)
        return min(self.hand.cards, key=lambda card: card.rank)

    def play_just_higher(self, game_state=None):
        """
        Plays the smallest rank card from hand that exceeds all cards played so
        far in the trick. If no card exceeds those already played, calls
        play_lowest_no_point to select a card.

        Args:
            game_state (dict, optional): The current state of the game.

        Returns:
            Card: selected card.
        """
        if game_state is None:
            game_state = {}

        played_ranks = [card.rank for card in game_state.get('trick', [])
                        if card is not None]
        highest_played_rank = max(played_ranks, default=0)

        # Find cards in hand that are just higher than the highest played card
        # special case for 2 and 3 of trumps, since they cannot catch any trumps
        just_higher_cards = [card for card in self.hand.cards
                             if card.rank > highest_played_rank and card.rank not in [2, 3]]
        if just_higher_cards:
            return min(just_higher_cards, key=lambda card: card.rank)

        return self.play_lowest_no_point(game_state=game_state)

    def play_lowest_no_point(self, game_state=None):
        """
        Plays the lowest value card while prioritizing the 2 of trumps and avoiding point loss.

        Args:
            game_state (dict, optional): The current state of the game. Should contain
                the key 'trumps' (str) for trump suit identification. Defaults to None.

        Returns:
            Card: The selected card following this priority order:
                1. The 2 of trumps if present in hand
                2. Lowest ranked trump card with zero point value
                3. Single-point trump cards (avoiding the 3 if possible)
                4. Lowest trump card by point value if only high-point trumps remain
                5. Lowest ranked "off" card if no trump cards available
        """
        if game_state is None:
            game_state = {}

        trump_cards = [card for card in self.hand.cards
                       if card.is_trump(suit=game_state.get('trumps', None))]

        # Try to play the 2 of trumps if present
        two_of_trumps = [card for card in trump_cards if card.rank == 2]
        if two_of_trumps:
            return two_of_trumps[0]

        # Play lowest trump card with zero point value
        zero_point_trumps = [card for card in trump_cards if card.points == 0]
        if zero_point_trumps:
            return min(zero_point_trumps, key=lambda card: card.rank)

        # Play trump card with lowest point value
        if trump_cards:
            # avoid playing the 3 if possible
            single_point_cards = [
                card for card in trump_cards if card.points == 1]
            if single_point_cards:
                return min(single_point_cards, key=lambda card: card.rank)

            return min(trump_cards, key=lambda card: card.points)

        # If no trumps, return "off" card
        return min(self.hand.cards, key=lambda card: card.rank)

    def play_single_point(self, game_state=None):
        """
        Plays a card worth a single point. If the 2 is the only available single
        point card, plays it. Otherwise, chooses the single point card with the
        lowest rank. Avoid playing the Ace (rank 17) if possible.
        
        If no point cards are available and trumps led, calls
        play_lowest_no_point. 
        
        If no point cards and trumps did not lead, calls
        play_off.

        Args:
            game_state (dict, optional): The current state of the game. Should
                contain the key 'trumps_led' (bool).

        Returns:
            Card: The card played by the player.
        """
        if game_state is None:
            game_state = {}

        # play any "catchable" single point cards
        # avoid playing the 2 or Ace since they are uncatchable
        single_point_cards = [card for card in self.hand.cards
                              if card.points == 1 and card.rank not in [2, 17]]
        if single_point_cards:
            return min(single_point_cards, key=lambda card: card.rank)

        # If have the 2, play it
        two_cards = [card for card in self.hand.cards if card.rank == 2]
        if two_cards:
            return two_cards[0]

        trumps_led = game_state.get('trumps_led', False)
        if trumps_led:
            trump_cards = [card for card in self.hand.cards
                           if card.is_trump(suit=game_state.get('trumps', None))]
            no_point_cards = [card for card in trump_cards if card.points == 0]
            if no_point_cards:
                return min(no_point_cards, key=lambda card: card.rank)

            # if we get here and have trumps, they must be the 3 and/or Ace
            # so play the lowest trump card
            if trump_cards:
                return min(trump_cards, key=lambda card: card.rank)

        # play "off" card if no single point cards or trumps
        return min(self.hand.cards, key=lambda card: card.rank)

    def play_save_point(self, game_state=None):
        """
        Plays a "catchable" single point card from the player's hand that ranks higher 
        than the highest card played in the trick. Excludes the 2 and Ace (rank 17)
        from consideration as they are uncatchable.
        If no such card exists, calls play_off to select a card.

        Args:
            game_state (dict, optional): The current state of the game. 
            Should contain 'trick' (list of Card objects).

        Returns:
            Card: The card played by the player.
        """
        if game_state is None:
            game_state = {}

        played_ranks = [card.rank for card in game_state.get('trick', [])
                        if card is not None]
        highest_played_rank = max(played_ranks, default=0)

        # find "catchable" single point cards higher than the highest played card
        single_point_cards = [card for card in self.hand.cards
                              if card.points == 1 and
                              card.rank not in [2, 17] and
                              card.rank > highest_played_rank]
        if single_point_cards:
            return min(single_point_cards, key=lambda card: card.rank)
        return self.play_off(game_state=game_state)

    def play_off(self, game_state=None):
        """
        Plays an "off" card (rank 0) if trumps were not led and such cards are available.
        Otherwise, plays the lowest ranked non-point card. If only point cards remain,
        avoids playing the 3 of trumps if possible and plays the lowest single-point card.
        As a last resort, plays the lowest ranked card available.

        Args:
            game_state (dict, optional): The current state of the game. Should contain
                the key 'trumps_led' (bool). Defaults to None.

        Returns:
            Card: The lowest appropriate card played by the player.
        """
        if game_state is None:
            game_state = {}

        trumps_led = game_state.get('trumps_led', False)

        # if trumps did not lead, play an "off" card (rank less than the 2)
        non_trump_cards = [card for card in self.hand.cards if card.rank < 2]
        if non_trump_cards and not trumps_led:
            # print(f"{self.name} playing off card {non_trump_cards[0]}")
            return non_trump_cards[0]

        # # If no non-trump cards or trumps led,
        # # If have the 2, play it
        two_cards = [card for card in self.hand.cards if card.rank == 2]
        if two_cards:
            return two_cards[0]

        # play the lowest ranked card that is not a point card
        lowest_no_point_cards = [
            card for card in self.hand.cards if card.points == 0]
        if lowest_no_point_cards:
            return min(lowest_no_point_cards, key=lambda card: card.rank)

        # If only have point cards, avoid playing the 3 if possible
        single_point_cards = [
            card for card in self.hand.cards if card.points == 1]
        if single_point_cards:
            return min(single_point_cards, key=lambda card: card.rank)

        # If all else fails, play the lowest ranked card
        return min(self.hand.cards, key=lambda card: card.rank)



# Remaining code is for command line argument parsing and main function
def get_args():
    """
    Parses command line arguments for player configuration and game modes.
    
    Configures argument parser with options for controlling game behavior,
    output verbosity, training modes, and configuration file loading.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - hands (int): Number of hands to play (default: 1)
            - verbose (bool): Enable verbose output flag
            - train (bool): Enable training mode flag  
            - debug (bool): Enable debug output flag
            - config (str): Path to YAML configuration file
            - interactive (bool): Enable interactive mode flag
    """
    parser = argparse.ArgumentParser(description='Pitch Player')

    # Add command line arguments
    parser.add_argument('--hands', type=int, default=1,
                        help='Number of hands to play')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Enable training mode')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Enable interactive mode')

    # Parse command line arguments
    args = parser.parse_args()

    return args

def load_config(config_file=None):
    """
    Load a YAML configuration file.

    Args:
        config_file (str, optional): The path to the YAML configuration file. 
            Defaults to None.

    Returns:
        dict: A dictionary containing the configuration settings, or an empty
            dictionary if no config file is provided.
    """
    config = {}
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        config = dict(yaml_data)  # Convert YAML to Python dictionary

    return config


def configure_players(config, game_state):
    """
    Creates and configures four player objects based on YAML configuration data.
    
    Dynamically imports player modules and classes, supports different player types
    and strategies per position. Destructively modifies the config dictionary by
    removing player configurations as they are processed.

    Args:
        config (dict): Configuration dictionary containing player settings. 
            Expected to have player1, player2, player3, player4 keys with:
            - module (str): Python module name (defaults to 'player')
            - class (str): Class name within module (defaults to 'Player')  
            - name (str): Display name (defaults to 'Player N')
            - strategies (dict): Strategy configuration dictionary
            Additional key-value pairs become player attributes.
            This dict is modified by removing processed player configs.
        game_state (dict): Game state dictionary used for verbose output control.
            Should contain 'verbose' (bool) key.

    Returns:
        list[Player]: List of 4 configured player objects in position order.
    """
    players = []
    for i in range(4):
        player_config = config.pop(f'player{i+1}', {})

        module = importlib.import_module(player_config.pop('module', 'player'))
        player_class = getattr(module, player_config.pop('class', 'Player'))
        new_player = player_class(name=player_config.pop('name', f'Player {i+1}'),
                                  strategies=player_config.pop('strategies', None),
                                  **player_config)

        players.append(new_player)
        if game_state.get('verbose', False):
            print(f"Player {i+1}: {new_player:<15}")
            pprint(new_player.state())

    return players

def main():
    """
    Demonstrates comprehensive Player class functionality through simulated gameplay.
    
    This main function showcases the complete player workflow including command line
    argument processing, YAML configuration loading, dynamic player creation, and
    simulated bidding rounds with trump selection and hand management.

    Workflow:
        1. Parses command line arguments for game configuration
        2. Loads optional YAML configuration file for player setup
        3. Creates game state dictionary from command line flags
        4. Configures four players using dynamic module loading
        5. Simulates specified number of hands with:
           - Deck creation, reset, and shuffling
           - Hand dealing (9 cards per player)
           - Player bidding with strategy application
           - Trump selection and hand sorting
           - Non-trump card discarding
           - Hand refilling to 6 cards from deck
           - Final trump-only hand display

    Features Demonstrated:
        - Command line argument integration
        - YAML configuration file processing
        - Dynamic player class instantiation
        - Strategy pattern implementation
        - Bidding logic with personality factors
        - Trump-based card filtering and hand management
        - Interactive mode for step-by-step review

    Output:
        Displays round separators, initial hands, bidding results with chosen
        trump suits, trump-only hands after discarding, and final hands after
        deck drawing. Verbose mode shows detailed player configurations.
    """
    # Parse command line arguments
    args = get_args()

    # Load configuration file
    config = load_config(args.config)

    # Set game state based upon command line arguments
    game_state = {}
    game_state['debug'] = args.debug
    game_state['verbose'] = args.verbose
    game_state['train'] = args.train
    game_state['interactive'] = args.interactive

    # Create player objects based on configuration
    players = configure_players(config, game_state=game_state)

    # Create a deck object
    deck = Deck()

    # demonstrate multiple bids
    for i in range(args.hands):
        print(f"{'-'*40}\nRound: {i}")

        deck.reset()
        deck.shuffle()
        hands = Hand(deck.deal(nhands=4, ncards=9))

        for i, next_hand in enumerate(hands):
            # Set the player's cards
            players[i].hand.replace_cards(next_hand)
            players[i].hand.sort_by_suit_and_rank()
            print(f'Player {players[i].name:<15}: {players[i].hand}')

        for player in players:
            bid = player.bid(game_state=game_state)
            trumps = player.choose_trumps()
            print(f'{player.name} bid {bid} in {trumps}')

            player.hand.set_trump(trumps)
            player.hand.sort_by_rank()
            player.hand.discard_non_trumps(trumps)
            print(f'\tTrumps: {player.hand}')

        if args.interactive:
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
