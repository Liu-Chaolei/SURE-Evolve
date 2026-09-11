import { access, readFile } from "node:fs/promises";
import { extname, join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { commands } from "../src/data/commands.ts";

const root = resolve(import.meta.dirname, "..");
const dist = join(root, "dist");
const pages = ["index.html", "docs/index.html", "404.html"];
const removedPages = ["workflows/index.html", "reliability/index.html"];

async function html(path: string): Promise<string> {
  return readFile(join(dist, path), "utf8");
}

function internalLinks(content: string): string[] {
  return [...content.matchAll(/href="([^"#]+)(?:#[^"]*)?"/g)]
    .map((match) => match[1])
    .filter((href): href is string => href !== undefined && href.startsWith("/") && !href.startsWith("//"));
}

function targetFor(href: string): string {
  const [clean = href] = href.split("?");
  if (clean.endsWith("/")) return join(dist, clean, "index.html");
  return extname(clean) ? join(dist, clean) : join(dist, `${clean}.html`);
}

describe("generated site", () => {
  it.each(pages)("renders an accessible, described page: %s", async (page) => {
    const content = await html(page);
    expect(content).toContain('<html lang="en">');
    expect(content).toContain('href="#main-content"');
    expect(content).toContain('<meta name="description"');
    expect(content).toContain('<meta property="og:title"');
    expect(content).toContain('<title>');
    expect((content.match(/<h1(?:\s|>)/g) ?? []).length).toBe(1);
  });

  it("marks the 404 page noindex", async () => {
    expect(await html("404.html")).toContain('<meta name="robots" content="noindex"');
  });

  it.each(removedPages)("does not emit removed route: %s", async (page) => {
    await expect(access(join(dist, page))).rejects.toThrow();
  });

  it("does not link to removed routes", async () => {
    for (const page of pages) {
      const content = await html(page);
      expect(content).not.toMatch(/href="\/(?:workflows|reliability)\//);
    }
  });

  it("uses consistent public repository setup instructions", async () => {
    const content = await html("docs/index.html");
    expect(content).toContain("git clone https://github.com/daqige/XLab.git");
    expect(content).toContain("cd XLab &amp;&amp; npm install --ignore-scripts");
    expect(content).not.toContain("earendil-works/pi");
  });

  it("ships the homepage walkthrough as progressively enhanced static HTML", async () => {
    const content = await html("index.html");
    expect(content).toContain('class="hero-terminal hero-walkthrough"');
    expect(content).toContain('class="pipeline-rail"');
    expect(content).toContain('class="run-card"');
    expect(content).toContain('class="artifact-lineage"');
    expect(content).toContain('class="trust-panels"');
    expect(content).toContain('class="ambient-control"');
    expect(content).toContain("data-loop-motion");
    expect(content).toContain('role="progressbar"');
    expect(content).toContain('aria-valuenow="72"');
    expect(content).toContain("ambientMotion");
    expect(content).toContain("IntersectionObserver");
    expect(content).toContain("prefers-reduced-motion: reduce");
    expect(content).toContain('class="progress-sheen"');
    expect(content).toContain("Research agents you can resume, inspect, and verify.");
    expect(content).toContain("npm install --ignore-scripts");
    expect(content).toContain("./pi-test.sh");
    expect(content).not.toMatch(/<(?:img|source)[^>]+(?:src|srcset)="https?:\/\//);
  });

  it("renders the independent docs shell and complete command reference", async () => {
    const content = await html("docs/index.html");
    expect(content).toContain('class="docs-shell"');
    for (const anchor of ["overview", "quickstart", "installation", "configuration", "core-concepts", "command-reference", "security", "source-reference"]) {
      expect(content).toContain(`id="${anchor}"`);
    }
    for (const command of commands) {
      expect((content.match(new RegExp(`id="${command.anchor}"`, "g")) ?? []).length).toBe(1);
    }
  });

  it("documents persistence, model separation, and execution boundaries", async () => {
    const content = await html("docs/index.html");
    for (const path of [".xlab/runs/", ".xlab/workspaces/", ".xlab/artifacts/"]) {
      expect(content).toContain(path);
    }
    expect(content).toContain("independent from the Pi Agent runtime model");
    expect(content).toContain("process-local");
    expect(content).toContain("Use trusted repositories");
    expect(content).toContain("external container, virtual machine, or other sandbox");
  });

  it("keeps cube-X brand assets synchronized", async () => {
    const [favicon, socialCard] = await Promise.all([
      readFile(join(dist, "favicon.svg"), "utf8"),
      readFile(join(dist, "social-card.svg"), "utf8"),
    ]);
    for (const asset of [favicon, socialCard]) {
      expect(asset).toContain("M12 17 32 8l20 9v30L32 56l-20-9Z");
      expect(asset).toContain("M52 47 32 32");
      expect(asset).toContain("#d38a4a");
    }
  });

  it("resolves every root-relative link and asset", async () => {
    for (const page of pages) {
      const content = await html(page);
      for (const href of internalLinks(content)) {
        await expect(access(targetFor(href)), `${page} links to missing ${href}`).resolves.toBeUndefined();
      }
    }
  });
});
