-- joss-author.lua
-- Pandoc Lua filter to extract author names and affiliations from
-- JOSS-structured YAML metadata into pandoc's default LaTeX template.

function Meta(meta)
  if not meta.authors then return meta end

  -- Build affiliation index -> name mapping
  local affil_map = {}
  if meta.affiliations then
    for _, a in ipairs(meta.affiliations) do
      local idx = pandoc.utils.stringify(a.index)
      affil_map[idx] = pandoc.utils.stringify(a["name"])
    end
  end

  -- Build author lines with superscript affiliation numbers
  local parts = {}
  for _, author in ipairs(meta.authors) do
    local name = pandoc.utils.stringify(author.name)
    local aff = pandoc.utils.stringify(author.affiliation or "")
    if aff ~= "" then
      table.insert(parts, name .. "\\textsuperscript{" .. aff .. "}")
    else
      table.insert(parts, name)
    end
  end
  local author_line = table.concat(parts, " \\quad ")

  -- Build affiliation lines
  local affil_lines = {}
  if meta.affiliations then
    for _, a in ipairs(meta.affiliations) do
      local idx = pandoc.utils.stringify(a.index)
      local name = pandoc.utils.stringify(a["name"])
      table.insert(affil_lines,
        "\\textsuperscript{" .. idx .. "}" .. name)
    end
  end

  if #affil_lines > 0 then
    author_line = author_line .. " \\\\[6pt]\n\\small " ..
      table.concat(affil_lines, " \\\\\n\\small ")
  end

  meta.author = pandoc.MetaInlines{pandoc.RawInline('latex', author_line)}
  return meta
end
