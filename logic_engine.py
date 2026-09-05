class KnowledgeBase:
    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string):
        """Assert a single fact into the KB."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Add a Horn Clause: premise_list (AND'd together) => conclusion_string."""
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        self.facts = set()

    def forward_chain(self):
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion in self.facts:
                    continue  
                if all(premise in self.facts for premise in premises):
                    self.facts.add(conclusion)
                    new_facts_added = True

        return self.facts


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()
    print("Facts after chaining:", kb.facts)