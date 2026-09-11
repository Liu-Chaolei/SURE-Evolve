import type { AutocompleteItem } from "@earendil-works/pi-tui";
import type { XlabCommandArgumentHint } from "./types.ts";

interface ArgumentToken {
	value: string;
	start: number;
	quote?: "'" | '"';
	wrappingQuote?: "'" | '"';
}

interface TokenizedArguments {
	tokens: ArgumentToken[];
	trailingWhitespace: boolean;
}

export interface XlabChoiceContext {
	argumentPrefix: string;
	valuePrefix: string;
}

function tokenizeArguments(input: string): TokenizedArguments {
	const tokens: ArgumentToken[] = [];
	let start: number | undefined;
	let value = "";
	let quote: ArgumentToken["quote"];
	let wrappingQuote: ArgumentToken["wrappingQuote"];
	let escaped = false;

	const finishToken = (): void => {
		if (start === undefined) {
			return;
		}
		tokens.push({ value, start, quote, wrappingQuote });
		start = undefined;
		value = "";
		quote = undefined;
		wrappingQuote = undefined;
		escaped = false;
	};

	for (let index = 0; index < input.length; index += 1) {
		const character = input[index] ?? "";
		if (start === undefined) {
			if (/\s/.test(character)) {
				continue;
			}
			start = index;
		}

		if (escaped) {
			value += character;
			escaped = false;
			continue;
		}
		if (character === "\\" && quote !== "'") {
			escaped = true;
			continue;
		}
		if (character === "'" || character === '"') {
			if (!quote) {
				quote = character;
				wrappingQuote ??= character;
				continue;
			}
			if (quote === character) {
				quote = undefined;
				continue;
			}
		}
		if (!quote && /\s/.test(character)) {
			finishToken();
			continue;
		}
		value += character;
	}

	const trailingWhitespace = start === undefined;
	finishToken();
	return { tokens, trailingWhitespace };
}

function completionValue(prefix: string, replacement: string, quote?: ArgumentToken["wrappingQuote"]): string {
	return `${prefix}${quote ?? ""}${replacement}${quote ?? ""} `;
}

function uniqueChoices(choices: AutocompleteItem[]): AutocompleteItem[] {
	const unique = new Map<string, AutocompleteItem>();
	for (const choice of choices) {
		const key = choice.value.toLowerCase();
		const previous = unique.get(key);
		unique.set(key, previous ? { ...previous, ...choice, value: previous.value } : choice);
	}
	return [...unique.values()];
}

function argumentChoices(
	argument: XlabCommandArgumentHint,
	argumentPrefix: string,
	valuePrefix: string,
	resolveChoices?: (argument: XlabCommandArgumentHint, context: XlabChoiceContext) => AutocompleteItem[] | undefined,
): AutocompleteItem[] {
	return uniqueChoices([
		...(argument.choices ?? []).map((choice): AutocompleteItem => ({ value: choice, label: choice })),
		...(resolveChoices?.(argument, { argumentPrefix, valuePrefix }) ?? []),
	]);
}

export function parseXlabArgumentTokens(input: string): string[] | undefined {
	const { tokens } = tokenizeArguments(input);
	return tokens.some((token) => token.quote) ? undefined : tokens.map((token) => token.value);
}

export function getXlabArgumentValue(input: string, name: string): string | undefined {
	const tokens = parseXlabArgumentTokens(input);
	if (!tokens) {
		return undefined;
	}
	for (let index = 0; index < tokens.length; index += 1) {
		const token = tokens[index] ?? "";
		if (token === name) {
			return tokens[index + 1];
		}
		if (token.startsWith(`${name}=`)) {
			return token.slice(name.length + 1);
		}
	}
	return undefined;
}

export function getXlabArgumentCompletions(
	arguments_: XlabCommandArgumentHint[] | undefined,
	argumentPrefix: string,
	resolveChoices?: (argument: XlabCommandArgumentHint, context: XlabChoiceContext) => AutocompleteItem[] | undefined,
): AutocompleteItem[] | null {
	if (!arguments_ || arguments_.length === 0) {
		return null;
	}

	const { tokens, trailingWhitespace } = tokenizeArguments(argumentPrefix);
	const current: ArgumentToken | undefined = trailingWhitespace
		? { value: "", start: argumentPrefix.length }
		: tokens[tokens.length - 1];
	if (!current) {
		return null;
	}

	const completedTokens = trailingWhitespace ? tokens : tokens.slice(0, -1);
	const replacementPrefix = argumentPrefix.slice(0, current.start);
	const byName = new Map(arguments_.map((argument) => [argument.name, argument]));

	const equalsIndex = current.value.indexOf("=");
	if (equalsIndex > 0) {
		const name = current.value.slice(0, equalsIndex);
		const argument = byName.get(name);
		if (!argument?.takesValue) {
			return null;
		}
		const rawValuePrefix = current.value.slice(equalsIndex + 1);
		const valuePrefix = rawValuePrefix.toLowerCase();
		const prefix = `${replacementPrefix}${name}=`;
		const choices = argumentChoices(argument, argumentPrefix, rawValuePrefix, resolveChoices).filter((choice) =>
			choice.value.toLowerCase().startsWith(valuePrefix),
		);
		return choices.length > 0
			? choices.map((choice) => ({
					...choice,
					value: completionValue(prefix, choice.value, current.wrappingQuote),
					description: choice.description ?? argument.description,
				}))
			: null;
	}

	const previous = completedTokens[completedTokens.length - 1];
	const valueArgument = previous ? byName.get(previous.value) : undefined;
	if (!current.value.startsWith("-") && valueArgument?.takesValue) {
		const valuePrefix = current.value.toLowerCase();
		const choices = argumentChoices(valueArgument, argumentPrefix, current.value, resolveChoices).filter((choice) =>
			choice.value.toLowerCase().startsWith(valuePrefix),
		);
		return choices.length > 0
			? choices.map((choice) => ({
					...choice,
					value: completionValue(replacementPrefix, choice.value, current.wrappingQuote),
					description: choice.description ?? valueArgument.description,
				}))
			: null;
	}

	if (current.wrappingQuote) {
		return null;
	}
	const assignmentPrefix = current.value.toLowerCase();
	const assignmentSuggestions = arguments_.filter(
		(argument) =>
			argument.syntax === "assignment" &&
			argument.name.toLowerCase().startsWith(assignmentPrefix) &&
			(argument.repeatable || !completedTokens.some((token) => token.value.startsWith(`${argument.name}=`))),
	);
	if (current.value !== "" && !current.value.startsWith("-") && assignmentSuggestions.length === 0) {
		return null;
	}

	const used = new Set(
		completedTokens.flatMap((token) => {
			const name = token.value.split("=", 1)[0];
			return name && byName.has(name) ? [name] : [];
		}),
	);
	const flagPrefix = current.value.toLowerCase();
	const suggestions = arguments_.filter(
		(argument) =>
			argument.syntax !== "assignment" &&
			argument.name.toLowerCase().startsWith(flagPrefix) &&
			(argument.repeatable || !used.has(argument.name)),
	);
	const candidates = current.value.startsWith("-")
		? suggestions
		: current.value === ""
			? [...suggestions, ...assignmentSuggestions]
			: assignmentSuggestions;
	return candidates.length > 0
		? candidates.map((argument) => {
				const assignment = argument.syntax === "assignment";
				const value = `${replacementPrefix}${argument.name}${assignment ? "=" : " "}`;
				return {
					value,
					label: argument.valueHint
						? `${argument.name}${assignment ? "=" : " "}${argument.valueHint}`
						: argument.name,
					description: argument.description,
				};
			})
		: null;
}

export function getXlabPositionalCompletions(
	argumentPrefix: string,
	resolve: (completed: string[], current: string) => AutocompleteItem[],
): AutocompleteItem[] | null {
	const { tokens, trailingWhitespace } = tokenizeArguments(argumentPrefix);
	const current: ArgumentToken | undefined = trailingWhitespace
		? { value: "", start: argumentPrefix.length }
		: tokens[tokens.length - 1];
	if (!current) {
		return null;
	}
	const completed = (trailingWhitespace ? tokens : tokens.slice(0, -1)).map((token) => token.value);
	const prefix = argumentPrefix.slice(0, current.start);
	const items = resolve(completed, current.value);
	return items.length > 0
		? items.map((item) => ({
				...item,
				value: completionValue(prefix, item.value, current.wrappingQuote),
			}))
		: null;
}
