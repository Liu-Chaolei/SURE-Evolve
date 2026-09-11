import type { ExtensionFactory } from "./extensions/index.ts";
import { xlabExtension } from "./xlab/index.ts";

const DEFAULT_EXTENSION_FACTORIES: ExtensionFactory[] = [xlabExtension];

export interface DefaultExtensionFactoryOptions {
	includeDefaults?: boolean;
}

export function withDefaultExtensionFactories(
	factories: ExtensionFactory[] = [],
	options: DefaultExtensionFactoryOptions = {},
): ExtensionFactory[] {
	const result = options.includeDefaults === false ? [] : [...DEFAULT_EXTENSION_FACTORIES];
	for (const factory of factories) {
		if (!result.includes(factory)) {
			result.push(factory);
		}
	}
	return result;
}
