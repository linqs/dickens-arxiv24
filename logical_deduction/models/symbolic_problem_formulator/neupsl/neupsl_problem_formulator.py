import json
import logging
import re

from models.symbolic_problem_formulator.base_symbolic_problem_formulator import BaseSymbolicProblemFormulator

DOMAIN = "Domain"
PREDICATES = "Predicates"
TARGETS = "Targets"
RULES = "Rules"
QUERY = "Query"
REWRITE = "Rewrite"

PROGRAM_OPTIONS = {
    "gurobi": {
        "options": {
            "runtime.log.level": "TRACE",
            "runtime.learn": "false",
            "runtime.inference": "true",
            "runtime.inference.method": "GurobiInference",
            "weightlearning.inference": "GurobiInference",
            "gurobi.worklimit": 60
        },
    },
    "no_gurobi": {
        "options": {
            "runtime.log.level": "TRACE",
            "runtime.learn": "false",
            "runtime.inference": "true",
            "admmreasoner.maxiterations": "200"
        },
    }
}

PREDICATE_OPTIONS = {
    "options": {
        "Integer": "true"
    }
}


class NeuPSLProblemFormulator(BaseSymbolicProblemFormulator):
    def __init__(self, arguments):
        super().__init__(arguments)

        self.query = None

    def formulate_symbolic_problem(self, response):
        domain, predicates, targets, rules, self.query = self.parse_response(response)
        rules = self.format_rules(rules, targets, domain)
        return self.create_neupsl_program(predicates, targets, rules)

    def parse_response(self, response):
        current_phase = ""

        domain = []
        predicates = []
        targets = {}
        rules = []
        query = []

        for line in response.splitlines():
            line = line.strip()
            if line == '':
                continue

            if line.replace(":", "") in [DOMAIN, PREDICATES, TARGETS, RULES, QUERY, REWRITE]:
                current_phase = line.replace(":", "")
                continue

            if current_phase == DOMAIN:
                domain = line.split(",")
                continue
            if current_phase == PREDICATES:
                predicates.extend(self._parse_predicates(line))
            elif current_phase == TARGETS:
                try:
                    predicate_targets = json.loads(line)
                except:
                    logging.warning("Invalid JSON: %s" % (line,))
                    continue

                for predicate_name in predicate_targets:
                    targets[predicate_name] = []
                    for variable_i in predicate_targets[predicate_name][0]:
                        for variable_j in predicate_targets[predicate_name][1]:
                            targets[predicate_name].append([variable_i, variable_j])
            elif current_phase == RULES:
                if line.startswith("//"):
                    continue
                rules.append(line.replace(" ", ""))
            elif current_phase == QUERY:
                if line.startswith("//"):
                    continue
                query_predicate = self._parse_predicates(line)[0]
                query.append((query_predicate[0].lower(), tuple(query_predicate[1])))
            elif current_phase == REWRITE:
                continue
            else:
                logging.warning("Not in a valid phase: %s" % (current_phase,))

        return domain, predicates, targets, rules, query

    def format_rules(self, rules, targets, domain):
        final_rules = []
        for rule in rules:
            final_rules.append(rule.replace("\"", "'") + " .")
            rule = rule.replace(" ", "")
            compare_operator_match = re.match(r"^(.*)->\((.*)([<>])(.*)\)$", rule)
            if compare_operator_match:
                final_rules.pop()

                rule_predicates = self._parse_predicates(compare_operator_match[1])
                if len(rule_predicates) != 2:
                    logging.warning("Invalid compare operator rule: %s\n" % (rule,))
                    continue

                less_variable = compare_operator_match[2]
                greater_variable = compare_operator_match[4]
                if compare_operator_match[3] == ">":
                    less_variable = compare_operator_match[4]
                    greater_variable = compare_operator_match[2]

                final_rules.extend(self._ground_compare_operator_rules(domain, targets, rule_predicates, less_variable, greater_variable))

        return final_rules

    def create_neupsl_program(self, predicates, targets, rules):
        if self.arguments.gurobi:
            neupsl_json = PROGRAM_OPTIONS["gurobi"].copy()
        else:
            neupsl_json = PROGRAM_OPTIONS["no_gurobi"].copy()
        neupsl_json["predicates"] = {}
        neupsl_json["rules"] = []

        for rule in rules:
            neupsl_json["rules"].append(rule)

        for predicate_name, predicate_arguments in predicates:
            neupsl_json["predicates"][predicate_name + "/" + str(len(targets[predicate_name][0]))] = PREDICATE_OPTIONS.copy()
            neupsl_json["predicates"][predicate_name + "/" + str(len(targets[predicate_name][0]))]["targets"] = targets[predicate_name]

        return neupsl_json

    def _parse_predicates(self, string):
        predicates = []
        string = string.replace(" ", "").replace("\"", "")
        while True:
            predicate_match = re.match(r"^(.*\(.*\))&(.*)$", string)
            if predicate_match is None:
                predicate_match = re.match(r"^(.*\(.*\))(.*)$", string)

            if not predicate_match:
                break

            predicate_name = predicate_match[1].split("(")[0]
            predicate_arguments = predicate_match[1].split("(")[1].replace(")", "").replace("'", "").split(",")
            predicates.append([predicate_name, predicate_arguments])
            string = predicate_match[2]

        return predicates

    def _ground_compare_operator_rules(self, domain, targets, rule_predicates, less_variable, greater_variable):
        """
        Ground compare operator rules assumes two predicates with the compare variables as the final arguments.
        :param targets: Dictionary of grounded targets. [targets = {"predicate name": [[arg1, ...] ...]}]
        :param rule_predicates: Two predicates involved in rule. [rule_predicates = ["predicate name", [arg1, ...]], ...]
        :param less_variable: Name of variable that is less.
        :param greater_variable: Name of variable that is greater.
        :return:
        """
        less_arguments = tuple()
        less_predicate_name = ""
        if less_variable == rule_predicates[0][-1][-1]:
            less_arguments = tuple(rule_predicates[0][-1][:-1])
            less_predicate_name = rule_predicates[0][0]
        elif less_variable == rule_predicates[1][-1][-1]:
            less_arguments = tuple(rule_predicates[1][-1][:-1])
            less_predicate_name = rule_predicates[1][0]
        else:
            logging.warning("Variable %s not found in final argument of predicates." % (less_variable,))

        greater_arguments = tuple()
        greater_predicate_name = ""
        if greater_variable == rule_predicates[0][-1][-1]:
            greater_arguments = tuple(rule_predicates[0][-1][:-1])
            greater_predicate_name = rule_predicates[0][0]
        elif greater_variable == rule_predicates[1][-1][-1]:
            greater_arguments = tuple(rule_predicates[1][-1][:-1])
            greater_predicate_name = rule_predicates[1][0]
        else:
            logging.warning("Variable %s not found in final argument of predicates." % (less_variable,))

        rules = []
        for greater_target in targets[greater_predicate_name]:
            if greater_arguments != tuple(greater_target[:-1]) or greater_target[-1] not in domain:
                continue
            if less_predicate_name not in targets:
                continue
            rules.append(greater_predicate_name + "('" + "','".join(greater_target) + "') <= 1")
            for less_target in targets[less_predicate_name]:
                if less_arguments != tuple(less_target[:-1]) or less_target[-1] not in domain:
                    continue
                if domain.index(less_target[-1]) <= domain.index(greater_target[-1]):
                    continue
                rules[-1] += " - " + less_predicate_name + "('" + "','".join(less_target) + "')"
            rules[-1] += " ."
        return rules
