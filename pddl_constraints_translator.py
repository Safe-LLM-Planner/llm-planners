import re
from sexpdata import loads, Symbol, dumps

class PDDLConstraintsTranslator:
    """
    A class responsible for translating a PDDL domain by incorporating constraints from a problem file.
    """

    @staticmethod
    def extract_constraints_from_problem(problem_text: str) -> str:
        """
        Extracts the constraints block from a PDDL problem and formats it as a PDDL string.

        Args:
            problem_text (str): PDDL problem.

        Returns:
            str: The inner content of the :constraints block formatted as a PDDL string,
                 or an empty string if not found.
        """
        
        try:
            # Parse the PDDL problem text into a nested S-expression
            parsed = loads(problem_text)
        except Exception as e:
            raise ValueError(f"Failed to parse PDDL: {e}")
        
        # Locate the :constraints block
        for element in parsed:
            if isinstance(element, list) and element[0] == Symbol(':constraints'):
                # Convert back to PDDL format, skipping the first element (:constraints)
                return dumps(element[1:])[1:-1].strip()  # Strip enclosing square brackets
        
        return ""  # No :constraints block found

    @staticmethod
    def _modify_action(action_block):
        """
        Modifies the preconditions and effects of an action block to include constraints management.

        Args:
            action_block (str): The PDDL action block as a string.

        Returns:
            str: The modified action block.
        """
        precondition_match = re.search(r':precondition\s*\(and\s*(.*?)\)', action_block, re.DOTALL)
        if precondition_match:
            preconditions = precondition_match.group(1)
            modified_preconditions = f'(constraints-have-been-checked) {preconditions}'
            action_block = re.sub(
                r':precondition\s*\(and\s*.*?\)', 
                f':precondition (and {modified_preconditions})', 
                action_block, 
                flags=re.DOTALL
            )
        
        effect_match = re.search(r':effect\s*\(and\s*(.*?)\)', action_block, re.DOTALL)
        if effect_match:
            effects = effect_match.group(1)
            modified_effects = f'(not (constraints-have-been-checked)) {effects}'
            action_block = re.sub(
                r':effect\s*\(and\s*.*?\)', 
                f':effect (and {modified_effects})', 
                action_block, 
                flags=re.DOTALL
            )
        
        return action_block

    def translate_domain(self, domain_text, problem_text) -> str:
        """
        Translates a PDDL domain by incorporating constraints from a PDDL problem.

        Args:
            domain_text (str): Input PDDL domain.
            problem_text (str): Input PDDL problem.
        """
        constraints = self.extract_constraints_from_problem(problem_text)
        if not constraints:
            raise ValueError("No constraints found in problem text")

        # Add the constraints predicate
        domain_text = domain_text.replace(
            '(:predicates', 
            '(:predicates (constraints-have-been-checked)\n'
        )

        # Add the verify-constraints action
        verify_constraint_action = f"""
        (:action verify-constraints
            :precondition (and (not (constraints-have-been-checked))
            {constraints})
            :effect (and (constraints-have-been-checked))
        )"""
        domain_text = re.sub(
            r'(\(:action \w+\s*(?:.|\n)*?:effect.*?\))', 
            lambda m: self._modify_action(m.group(1)), 
            domain_text
        )

        # Append the new action to the domain
        last_paren_index = domain_text.rfind(')')  # Find the last occurrence of ')'
        if last_paren_index != -1:  # Ensure that ')' exists in the string
            domain_text = domain_text[:last_paren_index] + f'{verify_constraint_action})' + domain_text[last_paren_index+1:]

        return domain_text

    def translate_plan_back(self, plan_text: str) -> str:
        """
        Removes all 'verify-constraints' actions from the translated plan.

        Args:
            plan_text (str): The original plan as a multiline string.

        Returns:
            str: The cleaned plan with 'verify-constraints' actions removed.
        """
        # Split the plan into lines
        lines = plan_text.strip().splitlines()

        # Filter out lines that contain only the 'verify-constraints' action
        filtered_lines = [
            line for line in lines if not line.strip().startswith("(verify-constraints")
        ]

        # Join the filtered lines back into a single string
        return "\n".join(filtered_lines)
