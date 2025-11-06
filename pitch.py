#!/usr/bin/env python
"""
Pitch Module

This module defines the Pitch class, which orchestrates complete games and matches 
of the Pitch card game. The module manages all aspects of gameplay including player
initialization, bidding phases, trick-taking, scoring with set detection, and 
match statistics. Supports configurable game modes, interactive play, and 
comprehensive game state tracking for AI development.

Classes:
    Pitch: The main game controller managing players, game flow, and scoring.

Functions:
    get_args(): Parses command line arguments for game configuration.
    load_config(config_file): Loads YAML configuration files for player setup.
    configure_players(config, game_state): Creates configured player objects.
    main(): Main entry point demonstrating complete match functionality.

Features:
    - Complete Pitch game implementation with standard rules
    - 4-player team-based gameplay (teams A: 0,2 and B: 1,3)
    - Comprehensive bidding phase with dealer forced-bid rules
    - Trump-based trick taking with "out" player detection
    - Pitch-specific scoring including set penalties and special 2-of-trumps rules
    - Match statistics with winning percentages
    - Interactive mode for step-by-step gameplay
    - Configurable player strategies via YAML files
    - Game state tracking for AI training and analysis

Programmer: Michelle Talley
Copyright (c) 2025 Michelle Talley
"""

import time
import importlib
import argparse
import random
from types import MappingProxyType
#from pprint import pprint
import yaml

from card import Card
import deck
import player


class Pitch:
    """
    The main game controller for complete Pitch card game matches.
    
    Manages all aspects of Pitch gameplay from player initialization through
    match completion, including bidding phases, trick-taking, scoring with
    set detection, and comprehensive game state tracking. Supports both
    single games and multi-game matches with detailed statistics.

    Attributes:
        deck (Deck): The card deck used for dealing and game management.
        players (list[Player]): Four Player objects in positions 0-3.
        teams (dict): Team assignments {'A': [0, 2], 'B': [1, 3]}.
        game_state (dict): Comprehensive game state dictionary containing all
            current game information. See detailed documentation below.

    Team Structure:
        - Team A: Players in positions 0 and 2 (North/South)
        - Team B: Players in positions 1 and 3 (East/West)
        - Each player has a 'team' attribute set during initialization

    Core Methods:
        __init__(players, **kwargs): Initializes game with players and settings.
        game(): Runs a complete single game to 31 points.
        match(games): Runs multiple games with match statistics.
        
    Game Flow Methods:
        deal_hand(): Deals initial 9-card hands to all players.
        bid_phase(dealer_index): Manages bidding with dealer forced-bid rule.
        fill_hands(suit): Completes hands after trump selection.
        play_hand(lead): Manages all 6 tricks of a hand.
        
    Scoring Methods:
        trick_winner(lead): Determines trick winner using card comparison.
        trick_points(winner): Calculates points with special 2-of-trumps rule.
        count_points(lead): Combined trick winner and point calculation.
        
    State Management:
        score_reset(): Resets game scores to 0-0.
        hand_reset(): Resets all hand-level state variables.
        trick_reset(lead): Resets trick-level state for new trick.

    Game State Documentation:
    
    The game_state dictionary contains comprehensive information passed to players
    as an immutable MappingProxyType. Updated after each player action.

    Core Game Settings:
        'debug' (bool): Enable detailed debug output
        'verbose' (bool): Enable verbose game information display
        'train' (bool): Enable training mode for AI development
        'interactive' (bool): Pause for user input between phases

    Scoring Information:
        'score' (dict): Current match scores {'A': int, 'B': int}
        'hand_points' (dict): Points scored in current hand {'A': int, 'B': int}

    Player and Bidding:
        'dealer' (int): Current dealer position (0-3)
        'current_bids' (list): Bids by position [int/None] (None=not bid, 0=pass)
        'bidder' (int): Position of winning bidder
        'bid' (int): Winning bid value

    Trump and Card Tracking:
        'trumps' (str): Current trump suit ('Spades', 'Hearts', 'Clubs', 'Diamonds')
        'trumps_remaining' (list): Unplayed trump ranks in descending order
        'trumps_played' (list): Played trump ranks
        'trumps_led' (bool): Whether trumps were led in current trick

    Trick Information:
        'lead' (int): Position of player who led current trick
        'trick_position' (int): Current player position relative to leader
        'trick' (list): Cards played in current trick by position [Card/None]
        'out' (list): Players known to be out of trumps [bool] by position

    """

    def __init__(self, players=None, **kwargs):
        """
        Initializes a new Pitch game instance with players and game settings.
        
        Sets up the game environment including deck creation, player assignment
        with position and team attributes, and comprehensive game state initialization.
        Creates default players with compass names if none provided.

        Args:
            players (list[Player], optional): List of exactly 4 Player objects.
                If None or empty, creates default players named North, East, 
                South, West. Defaults to None.
            **kwargs: Additional keyword arguments added to the game_state dictionary.
                Common keys include 'debug', 'verbose', 'train', 'interactive'.
        """
        self.deck = deck.Deck()

        if players or len(players) == 0:
            self.players = players
        else:
            self.players = [player.Player() for _ in range(4)]
            self.players[0].name = 'North'
            self.players[1].name = 'East'
            self.players[2].name = 'South'
            self.players[3].name = 'West'

        for i in range(4):
            self.players[i].position = i

        self.teams = {'A': [0, 2], 'B': [1, 3]}
        for member in self.teams['A']:
            self.players[member].team = 'A'
        for member in self.teams['B']:
            self.players[member].team = 'B'

        self.game_state = {}
        self.game_state.update(kwargs)  # Set additional keyword arguments in game_state
        self.score_reset()
        self.hand_reset()
        self.trick_reset()

    def score_reset(self):
        """
        Resets the match scores to 0-0 for both teams.
        
        Initializes the game_state['score'] dictionary with zero points
        for both Team A and Team B. Called at the start of each new game
        within a match.
        """
        self.game_state['score'] = {'A': 0, 'B': 0}

    def hand_reset(self):
        """
        Resets all hand-level game state variables for a new hand.
        
        Initializes bidding information, trump tracking, point accumulation,
        and player status for the start of a new hand. Called before each
        bidding phase within a game.
        """
        self.game_state['current_bids'] = [None, None, None, None]
        self.game_state['bidder'] = None
        self.game_state['bid'] = 0
        self.game_state['trumps'] = None
        self.game_state['hand_points'] = {'A': 0, 'B': 0}
        self.game_state['trumps_led'] = False
        self.game_state['trumps_played'] = []
        self.game_state['trumps_remaining'] = list(range(17, -1, -1))
        self.game_state['out'] = [False, False, False, False]

    def trick_reset(self, lead=0):
        """
        Resets trick-level game state variables for a new trick.
        
        Prepares the game state for the next trick by setting the leading
        player and clearing trick-specific information. Called before each
        of the 6 tricks within a hand.

        Args:
            lead (int, optional): Position (0-3) of the player who will lead
                this trick. Defaults to 0.
        """
        self.game_state['lead'] = lead
        self.game_state['trick_position'] = 0
        self.game_state['trick'] = [None, None, None, None]
        self.game_state['trumps_led'] = False

    def bid_phase(self, dealer_index):
        """
        Manages the complete bidding phase with Pitch-specific rules.
        
        Conducts bidding starting with the player after the dealer and proceeding
        clockwise. Implements the dealer forced-bid rule where the dealer must
        bid 5 if no one else has bid. Updates game state and provides formatted
        output of all bids and the final winner.

        Args:
            dealer_index (int): Position (0-3) of the current dealer.

        Returns:
            tuple[int, int]: A tuple containing:
                - bidder (int): Position of the winning bidder
                - current_bid (int): The winning bid value

        Bidding Rules:
            - Bidding starts with player after dealer (dealer_index + 1)
            - Players bid in clockwise order
            - Each player must bid higher than current bid or pass (bid 0)
            - If no one bids, dealer is forced to bid 5
            - All bids are recorded in game_state['current_bids']

        Output Format:
            Prints formatted bidding section with each player's bid/pass decision
            and announces the final winner with their bid amount.

        Game State Updates:
            Updates current_bids list with each player's bid value where
            None indicates not yet bid and 0 indicates pass.
        """
        current_bid = 0
        bidder = None

        print(f"Bidding {'-' * 60}")

        for i in (i % 4 for i in range(dealer_index + 1, dealer_index + 5)):

            # copy the game state for the player
            # game_state = self.game_state.copy()
            read_only_game_state = MappingProxyType(self.game_state)
            bid = self.players[i].bid(current_bidder=bidder,
                                      current_bid=current_bid,
                                      game_state=read_only_game_state)

            # if the dealer is the last to bid, the bid is automatically 5
            if current_bid == 0 and i == dealer_index:
                bid = 5
                current_bid = 5
                bidder = i
            elif bid > current_bid:
                current_bid = bid
                bidder = i

            self.game_state['current_bids'][i] = bid
            if bid == 0:
                print(f'\t{self.players[i].name} passes.')
            else:
                print(f'\t{self.players[i].name} bids {bid}.')

        print(f'\t{self.players[bidder].name} wins with {current_bid} bid.')
        return bidder, current_bid

    def trick_winner(self, lead=0):
        """
        Determines the winner of the current trick using card comparison.

        Args:
            lead (int, optional): Position of the player who led the trick.
                Used for initial winner assumption. Defaults to 0.

        Returns:
            int: Position (0-3) of the player who played the winning card.
        """
        # initially assume the lead player wins
        # if all players play off, the lead player wins trick
        winner = lead
        lead_card = self.game_state['trick'][lead]
        for j in range(0, 4):
            if not self.game_state['trick'][j]:
                continue
            if self.game_state['trick'][j] > lead_card:
                lead_card = self.game_state['trick'][j]
                winner = j
        return winner

    def trick_points(self, winner):
        """
        Calculates points awarded to each team from the current trick.
        
        Implements Pitch-specific scoring rules where most points go to the
        winning team, but the 2 of trumps points go to whichever team played it.

        Args:
            winner (int): Position of the player who won the trick.

        Returns:
            tuple[int, int]: Points awarded to (Team A, Team B).
        """
        winning_team = self.players[winner].team
        points = {'A': 0, 'B': 0}

        for i, next_card in enumerate(self.game_state['trick']):
            if not next_card:
                continue
            if next_card.rank == 2:  # special case for the 2 of trumps
                points[self.players[i].team] += next_card.points
            else:
                points[winning_team] += next_card.points
        return points['A'], points['B']

    def count_points(self, lead=0):
        """
        Combined trick winner determination and point calculation.
        
        Convenience method that determines the trick winner and calculates
        points in a single call. Provides all information needed to update
        game state after a trick is completed.

        Args:
            lead (int, optional): Position of the player who led the trick.
                Defaults to 0.

        Returns:
            tuple[int, int, int]: A tuple containing:
                - team1_points (int): Points awarded to Team A
                - team2_points (int): Points awarded to Team B  
                - winner (int): Position of the trick winner
        """
        team1_points = 0
        team2_points = 0

        # determine the winner of the trick
        winner = self.trick_winner(lead=lead)

        team1_points, team2_points = self.trick_points(winner)
        return team1_points, team2_points, winner

    def play_hand(self, lead=0):
        """
        Manages the complete play of a hand including all 6 tricks.
        
        Orchestrates the trick-taking phase of a hand, handling player turns,
        trump-led detection, out-of-trumps tracking, point calculation, and
        formatted output. Implements core Pitch rules for trick play and
        player elimination.

        Args:
            lead (int, optional): Position of the player who leads the first trick.
                Typically the winning bidder. Defaults to 0.

        Game Flow:
            1. Resets trick state for each of 6 tricks
            2. Players play cards in clockwise order from leader
            3. Tracks trump-led status and player "out" conditions
            4. Records played trumps and updates remaining trump lists
            5. Calculates trick points and updates hand totals
            6. Winner of trick leads the next trick

        Game State Updates:
            - Updates trick arrays with played cards
            - Maintains trump tracking lists
            - Accumulates hand_points for both teams
            - Updates player out status when detected
            - Sets trumps_led flag based on first card
        """
        print(f"Tricks {'-' * 60}")

        # play all 6 cards for each player (6 tricks or rounds)
        for _ in range(6):

            # Play a trick
            # circular loop through players starting with current lead
            # play_msg = 'Play: '
            play_msg = '\t'

            self.trick_reset(lead=lead)
            for j in [k % 4 for k in range(lead, lead + 4)]:
                # if player already out, skip to next player
                if self.game_state['out'][j]:
                    play_msg += f"{self.players[j].name:8}: out    "
                    self.game_state['trick_position'] = (
                        self.game_state['trick_position'] + 1) % 4
                    continue

                # copy the game state for the player
                read_only_game_state = MappingProxyType(self.game_state)

                next_card = self.players[j].play_card(game_state=read_only_game_state)

                if j == lead:
                    self.game_state['trumps_led'] = next_card.is_trump()
                else:
                    if self.game_state['trumps_led'] and \
                        next_card.is_nontrump(suit=self.game_state['trumps']):
                        self.game_state['out'][j] = True

                # if player is now out, report them as out
                if self.game_state['out'][j]:
                    play_msg += f"{self.players[j].name:8}: out    "
                else:
                    play_msg += f"{self.players[j].name:8}: {next_card}    "

                self.game_state['trick'][j] = next_card
                self.game_state['trick_position'] = (self.game_state['trick_position'] + 1) % 4

            # record all the cards played in the trick
            for played_card in self.game_state['trick']:
                if played_card is None:
                    continue
                if played_card.is_trump():
                    self.game_state['trumps_played'].append(played_card.rank)
                    self.game_state['trumps_remaining'].remove(played_card.rank)

            # calculate points from trick and assign to winning team
            team1_points, team2_points, winner = self.count_points(lead=lead)
            self.game_state['hand_points']['A'] += team1_points
            self.game_state['hand_points']['B'] += team2_points
            lead = winner

            print(f"{play_msg}"
                    f"(A: {team1_points} "
                    f"B: {team2_points}) "
                    f"Winner: {self.players[winner].name:10} "
                    f"({self.players[winner].team}) ")

        print(f"\tHand points: A = {self.game_state['hand_points']['A']} "
                f"B = {self.game_state['hand_points']['B']}")


    def deal_hand(self):
        """
        Deals initial 9-card hands to all players for bidding.
        
        Resets and shuffles the deck, then deals 9 cards to each of the 4 players.
        Replaces each player's hand with their new cards and sorts them by suit
        and rank for display during bidding. This is the initial dealing before
        trump selection and hand completion.
        """
        self.deck.reset()
        self.deck.shuffle()

        hands = self.deck.deal(nhands=4, ncards=9)
        for i, next_player in enumerate(self.players):
            next_player.hand.replace_cards(hands[i])
            next_player.hand.sort_by_suit_and_rank()

    def fill_hand(self, next_player, suit, is_bidder=False):
        """
        Fills a single player's hand with cards from the deck based upon trump suit. 
        
        For bidders, adds all remaining deck cards then discards non-trumps. 
        
        For non-bidders, first discards non-trumps then deals additional cards to 
        reach 6 cards.

        Args:
            next_player (Player): The player whose hand is being filled.
            suit (str): The trump suit.
            is_bidder (bool, optional): Whether this player is the bidder. 
                Defaults to False.
        """
        if is_bidder:
            next_player.hand.add_cards(self.deck.cards)
            next_player.hand.set_trump(suit)
            next_player.hand.discard_non_trumps(suit=suit, is_bidder=True)
        else:
            next_player.hand.set_trump(suit)
            next_player.hand.discard_non_trumps(suit=suit)
            need = 6 - next_player.hand.count()
            if need > 0:
                next_player.hand.add_cards(self.deck.deal(nhands=1, ncards=need)[0])
            next_player.hand.set_trump(suit)

        next_player.hand.sort_by_rank()


    def fill_hands(self, suit):
        """
        Completes all player hands to 6 cards after trump selection and discarding.
        
        After trump selection, players discard non-trump cards and need their hands
        refilled to exactly 6 cards for gameplay. Handles non-bidders first, then
        gives the bidder the remaining deck cards for advantage.

        Args:
            suit (str): The selected trump suit for determining hand completion.

        Bidder Advantage:
            The bidder receives cards last and gets all remaining deck cards,
            providing strategic advantage for making their bid. This implements
            the standard Pitch rule where the bidder gets the "widow" cards.

        Error Handling:
            May raise ValueError if insufficient cards remain in deck,
            which is caught by the calling game() method to trigger re-deal.
        """
        bidder = self.game_state['bidder']
        for i, next_player in enumerate(self.players):
            if i == bidder: # complete bidder's hand later
                continue

            self.fill_hand(next_player=next_player, suit=suit, is_bidder=False)

        # now complete the bidder's hand with the remaining deck
        self.fill_hand(next_player=self.players[bidder], suit=suit, is_bidder=True)

    def game(self):
        """
        Manages a complete single game of Pitch to 31 points.
        
        Orchestrates the full game flow from initial setup through final scoring,
        implementing all Pitch rules including set detection, dealer rotation,
        and proper game termination. Handles error recovery for invalid deals.

        Returns:
            str: The winning team identifier ('A' or 'B').

        Game Flow:
            1. Initializes game with score reset and random dealer selection
            2. Continues hands until a team reaches 31 points
            3. For each hand:
               - Deals initial 9-card hands
               - Conducts bidding phase
               - Handles trump selection and hand completion
               - Plays all 6 tricks
               - Calculates final scoring with set detection
               - Rotates dealer position
            4. Determines final winner with bidder advantage rule

        Output Format:
            Provides comprehensive game output including:
            - Game start/end markers
            - Dealer announcements
            - Bidding and trump selection results
            - Optional verbose hand displays
            - Scoring updates with set notifications
            - Final game result with winning team and score

        Interactive Mode:
            If interactive mode is enabled, pauses for user input between hands
            to allow step-by-step game review.
        """
        print(f"{'=' * 60}")
        print('Game Start')

        self.score_reset()
        dealer_index = random.randint(0, 3)

        while (self.game_state['score']['A'] < 31) and (self.game_state['score']['B'] < 31):

            self.hand_reset()
            self.deal_hand()

            print(f'Dealer: {self.players[dealer_index].name}')
            if self.game_state['verbose']:
                print(f"Bid Hands {'-' * 60}")
                for next_player in self.players:
                    print(f"\t{next_player.name:15} {next_player.hand}")

            bidder, bid = self.bid_phase(dealer_index=dealer_index)
            self.game_state['bidder'] = bidder
            self.game_state['bid'] = bid

            trumps = self.players[bidder].choose_trumps()
            self.game_state['trumps'] = trumps

            # complete the non-bidder players' hands
            try:
                self.fill_hands(suit=trumps)
            except ValueError as fill_hand_error:
                print(f"{fill_hand_error}")
                print("Deal again.")
                continue

            print(f'\t{self.players[bidder].name} bids {bid} in {trumps}.')

            if self.game_state['verbose']:
                print(f"Complete Hands {'-' * 60}")
                for next_player in self.players:
                    print(f"\t{next_player.name:15} {next_player.hand}")

            self.play_hand(lead=bidder)

            # scoring phase
            bidding_team = self.players[bidder].team

            # check if bidder went set and deduct bid points if necessary
            bid_result = f"made {self.game_state['hand_points'][bidding_team]} points "
            bid_result += f"on {self.game_state['bid']} bid"
            if self.game_state['hand_points'][bidding_team] < self.game_state['bid']:
                print(f"\t{self.players[bidder].name} went set ({bid_result})")
                self.game_state['hand_points'][bidding_team] = -self.game_state['bid']
            else:
                print(f"\t{self.players[bidder].name} made bid ({bid_result})")

            # accumulate total score
            for team in ('A', 'B'):
                self.game_state['score'][team] += self.game_state['hand_points'][team]

            print(f"Score: A: {self.game_state['score']['A']} "
                    f"B: {self.game_state['score']['B']}")

            # next dealer
            dealer_index = (dealer_index + 1) % 4

            if self.game_state.get('interactive', False):
                # wait for user input before continuing
                input("Press Enter to continue...")

        # determine the winner of the game; check edge case of both teams over 31 first
        if self.game_state['score']['A'] >= 31 and self.game_state['score']['B'] >= 31:
            winner = 'A' if bidder in (0, 2) else 'B'
        else:
            winner = 'A' if self.game_state['score']['A'] > self.game_state['score']['B'] else 'B'

        winning_team_players = [self.players[i].name for i in self.teams[winner]]

        print(f"Game result: Team {winner} ({' and '.join(winning_team_players)}) wins "
              f"with score {self.game_state['score'][winner]}")
        print('=' * 60)

        return winner

    def match(self, games=1):
        """
        Manages a complete match consisting of multiple games with statistics.
        
        Runs the specified number of games and provides comprehensive match
        statistics including win counts, percentages, and overall match winner
        determination. Handles interactive pausing between games when enabled.

        Args:
            games (int, optional): The number of games to play in the match.
                Must be positive integer. Defaults to 1.

        Interactive Mode:
            If interactive mode is enabled, pauses for user input between
            games to allow match-level review.

        """
        team_a = [self.players[i].name for i in self.teams['A']]
        team_b = [self.players[i].name for i in self.teams['B']]
        print(f"Match: Team A ({' and '.join(team_a)}) vs Team B ({' and '.join(team_b)})")

        team_a_wins = 0
        team_b_wins = 0
        for _ in range(games):
            winner = self.game()
            if winner == 'A':
                team_a_wins += 1
            elif winner == 'B':
                team_b_wins += 1

            if self.game_state.get('interactive', False):
                # wait for user input before continuing
                input("Press Enter to continue...")


        print(
            f"Match result: Team A ({' and '.join(team_a)}) wins: {team_a_wins} "
            f"({(team_a_wins / games) * 100:.2f}%) "
            f"Team B ({' and '.join(team_b)}) wins: {team_b_wins} "
            f"({(team_b_wins / games) * 100:.2f}%)"
        )
        if team_a_wins > team_b_wins:
            print(f"Team A ({' and '.join(team_a)}) wins the match!")
        elif team_b_wins > team_a_wins:
            print(f"Team B ({' and '.join(team_b)}) wins the match!")
        else:
            print("The match is a tie!")


    def _setup_test_hand_state(self, test_hand):
        """
        Sets up the game state for test hand scenario.
        
        Args:
            test_hand (dict): The test hand configuration.
            
        Returns:
            tuple: (bidder_index, bid, trump_suit)
        """
        self.hand_reset()

        bidder_tag = test_hand.get("bidder", "player1")
        bidder_index = int(bidder_tag[-1]) - 1
        self.game_state['bidder'] = bidder_index

        bid = test_hand.get("bid", 5)
        self.game_state['bid'] = bid

        trump_suit = test_hand.get("trumps", "Spades")
        self.game_state['trumps'] = trump_suit

        return bidder_index, bid, trump_suit

    def _setup_player_hands(self, test_hand, trump_suit):
        """
        Sets up predetermined hands for all players with duplicate validation.
        
        Args:
            test_hand (dict): The test hand configuration.
            trump_suit (str): The trump suit for the hand.
        """
        all_cards = []

        for tag, value in test_hand.items():
            if tag.startswith("player"):
                position = int(tag[-1]) - 1
                card_list = []

                for card_spec in value:
                    new_card = Card(card_spec[0], card_spec[1])
                    card_list.append(new_card)

                    # Check for duplicates
                    card_key = (card_spec[0], card_spec[1])
                    if card_key in all_cards:
                        raise ValueError(
                            f"Duplicate card found: {card_spec[0]} of {card_spec[1]} "
                            f"appears in multiple hands"
                        )
                    all_cards.append(card_key)

                # Setup player's hand
                self.players[position].hand.replace_cards(card_list)
                self.players[position].hand.set_trump(trump_suit)
                self.players[position].hand.sort_by_rank()

    def _validate_hand_sizes(self):
        """
        Validates that each player has exactly 6 cards.
        
        Raises:
            ValueError: If any player doesn't have exactly 6 cards.
        """
        for i, next_player in enumerate(self.players):
            card_count = next_player.hand.count()
            if card_count != 6:
                raise ValueError(
                    f"Player {next_player.name} (position {i}) has {card_count} cards, "
                    f"expected 6 cards"
                )

    def play_test_hand(self, test_hand):
        """
        Plays a test hand scenario using the specified test hand configuration.

        Args:
            test_hand (dict): The test hand configuration containing player hands.
        """
        # Check if test_hand is empty
        if not test_hand:
            raise ValueError("test_hand values must be set in the configuration file")

        # Setup game state and get key values
        bidder_index, bid, trump_suit = self._setup_test_hand_state(test_hand)

        # Setup all player hands with validation
        self._setup_player_hands(test_hand, trump_suit)

        # Display test hand information
        print(f"{'=' * 60}")
        print('Test Hand Start')
        print(f'Test Hand: {self.players[bidder_index].name} bid {bid} in {trump_suit}')

        if self.game_state.get('verbose', False):
            print(f"Test Hands {'-' * 60}")
            for next_player in self.players:
                print(f"\t{next_player.name:15} {next_player.hand}")

        # Validate hand sizes
        self._validate_hand_sizes()

        # Play the hand
        self.play_hand(lead=bidder_index)

        print('Test Hand Complete')
        print('=' * 60)


def get_args():
    """
    Parses command line arguments for Pitch game configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - games (int): Number of games to play in match (default: 1)
            - verbose (bool): Enable verbose output flag
            - train (bool): Enable training mode flag
            - debug (bool): Enable debug output flag  
            - config (str): Path to YAML configuration file
            - interactive (bool): Enable interactive mode flag
            - test_hand (str|None): Test hand scenario name or None if not used
    """
    parser = argparse.ArgumentParser(description='Pitch Game')

    # Add command line arguments
    parser.add_argument('-g', '--games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-t', '--train', action='store_true',
                        help='Enable training mode')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('-c', '--config', help='Path to configuration file')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Enable interactive mode')
    parser.add_argument('--test_hand', type=str, nargs='?', const='default_test_hand',
                        help='Use predetermined test hand scenario from config. '
                             'Optionally specify scenario name, defaults to "default_test_hand"')

    # Parse command line arguments
    args = parser.parse_args()

    return args


def load_config(config_file=None):
    """
    Loads a YAML configuration file for player and game setup.
    
    Reads and parses a YAML configuration file containing player definitions,
    strategy specifications, and game parameters. Returns empty dictionary
    if no config file is specified.

    Args:
        config_file (str, optional): Path to the YAML configuration file.
            If None, returns empty configuration. Defaults to None.

    Returns:
        dict: Configuration dictionary containing parsed YAML data, or empty
            dictionary if no config file provided.

    Configuration Format:
        Expected YAML structure includes player definitions (player1-player4)
        with module, class, name, and strategies specifications. Additional
        game-level settings may also be included.

    Error Handling:
        File reading errors are not caught here - they will propagate to caller.
        YAML parsing errors from yaml.safe_load() will also propagate upward.
    """
    config = {}
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        config = dict(yaml_data)  # Convert YAML to Python dictionary

    return config


def configure_players(config, game_state=None):
    """
    Creates and configures four player objects based on YAML configuration.
    
    Dynamically imports player modules and classes, supports different player
    types and strategies per position. Handles configuration extraction and
    validation with error checking for missing classes.

    Args:
        config (dict): Configuration dictionary containing player settings.
            Expected to have player1-player4 keys with:
            - module (str): Python module name (defaults to 'player')
            - class (str): Class name within module (defaults to 'Player')
            - name (str): Display name (defaults to 'Player N')
            - strategies (dict): Strategy configuration dictionary
            Additional key-value pairs become player attributes.
            This dict is modified by removing processed player configs.
        game_state (dict, optional): Game state dictionary for debug output
            control. Defaults to None.

    Returns:
        list[Player]: List of 4 configured player objects in position order.

    Raises:
        ValueError: If specified player class is not found in the module.
    """
    players = []
    for i in range(4):
        player_config = config.pop(f'player{i+1}', {})

        # load the player class from the specified module;
        # if not specified, use the default Player class from player module
        module = importlib.import_module(player_config.pop('module', 'player'))
        player_class = getattr(module, player_config.pop('class', 'Player'))
        if not player_class:
            raise ValueError(f"Class {player_config['class']} not found in {module.__name__}")

        # create a new player instance with the configuration
        new_player = player_class(name=player_config.pop('name', f'Player {i+1}'),
                                  strategies=player_config.pop('strategies', None),
                                  **player_config)

        players.append(new_player)
        if game_state.get('debug', False):
            pass
            # print(f"Player {i+1}: {new_player}")
            # pprint(new_player.state())

    return players




def main():
    """
    Main entry point demonstrating complete Pitch game functionality.
    
    Orchestrates the full application workflow from command line processing
    through match completion, showcasing all major components of the Pitch
    game system including configuration loading, player setup, and match execution.

    Usage Examples:
        python pitch.py --games 10 --verbose
        python pitch.py --config wizards_config.yaml --interactive
        python pitch.py --debug --train --games 100
        python pitch.py --config wizards_config.yaml --test_hand
        python pitch.py --config wizards_config.yaml --test_hand hearts_challenge

    Output:
        Provides comprehensive match output including game progression,
        statistics, and final results through the Pitch class methods.
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
    game_state['interactive'] = args.interactive  # Add interactive mode to game state

    # Create player objects based on configuration
    players = configure_players(config, game_state=game_state)

    # Create Pitch instance
    pitch = Pitch(players=players, **game_state)

    if args.test_hand:
        # Get test hand scenarios and determine which one to use
        test_hands = config.get("test_hands", {})
        if not test_hands:
            raise ValueError("No test_hands configuration found in config file")

        # Determine scenario name to use
        if args.test_hand == 'default_test_hand':
            scenario_name = config.get("default_test_hand", list(test_hands.keys())[0])
        else:
            scenario_name = args.test_hand

        # Validate scenario exists
        if scenario_name not in test_hands:
            available = ', '.join(test_hands.keys())
            raise ValueError(f"Test hand scenario '{scenario_name}' not found. "
                           f"Available scenarios: {available}")

        print(f"Using test hand scenario: '{scenario_name}'")
        test_hand = test_hands[scenario_name]
        pitch.play_test_hand(test_hand)
    else:
        start_time = time.time()
        pitch.match(games=args.games)
        end_time = time.time()
        print(f"Match execution time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
