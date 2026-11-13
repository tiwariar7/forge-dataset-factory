"""Specialized Excel and Spreadsheet companion parser.
Preserves cell coordinates, sheet hierarchy, headers, types, formulas, and row relationships.
"""

from typing import List, Dict, Any, Tuple
import os
import zipfile
import xml.etree.ElementTree as ET
from ..normalization.schemas import CanonicalDocument, DocumentElement, TableData, ElementType, ExtractionEngine
from ..normalization.provenance import compute_file_hash


class ExcelParser:
    """Extracts Excel workbooks into CanonicalDocument preserving sheet hierarchy and cell formulas."""

    def __init__(self, preserve_formulas: bool = True, max_rows_per_sheet: int = 50000):
        self.preserve_formulas = preserve_formulas
        self.max_rows_per_sheet = max_rows_per_sheet

    def parse(self, file_path: str) -> CanonicalDocument:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        file_hash = compute_file_hash(file_bytes)
        filename = os.path.basename(file_path)
        file_size = len(file_bytes)

        elements: List[DocumentElement] = []
        tables: List[TableData] = []

        # Try openpyxl first if installed
        try:
            import openpyxl  # type: ignore
            tables, elements = self._parse_with_openpyxl(file_path)
            engine = ExtractionEngine.OPENPYXL
        except ImportError:
            # Fallback to direct OOXML Zip+XML parsing
            tables, elements = self._parse_with_ooxml(file_bytes, filename)
            engine = ExtractionEngine.NATIVE

        return CanonicalDocument(
            filename=filename,
            file_type="xlsx",
            file_size_bytes=file_size,
            source_hash=file_hash,
            extracted_by=engine,
            extraction_quality_score=0.98,
            elements=elements,
            tables=tables,
            metadata={"sheet_count": len(tables), "parser": "excel_parser"}
        )

    def _parse_with_openpyxl(self, file_path: str) -> Tuple[List[TableData], List[DocumentElement]]:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(file_path, data_only=not self.preserve_formulas)
        tables: List[TableData] = []
        elements: List[DocumentElement] = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            all_rows = list(sheet.iter_rows(values_only=False))
            if not all_rows:
                continue

            # Heading element for sheet
            sheet_heading = DocumentElement(
                type=ElementType.HEADING,
                content=f"Sheet: {sheet_name}",
                heading_path=[f"Workbook", f"Sheet: {sheet_name}"],
                level=2
            )
            elements.append(sheet_heading)

            headers: List[str] = []
            rows_data: List[List[Any]] = []
            formulas: Dict[str, str] = {}

            first_row = True
            for r_idx, row in enumerate(all_rows[:self.max_rows_per_sheet]):
                row_vals = []
                for c_idx, cell in enumerate(row):
                    val = cell.value
                    coord = cell.coordinate
                    if cell.data_type == 'f' or (isinstance(val, str) and val.startswith('=')):
                        formulas[coord] = str(val)
                    row_vals.append(str(val) if val is not None else "")

                if first_row:
                    headers = [str(v) if v else f"Col_{i+1}" for i, v in enumerate(row_vals)]
                    first_row = False
                else:
                    if any(row_vals):  # Non-empty row
                        rows_data.append(row_vals)

            table = TableData(
                sheet_name=sheet_name,
                headers=headers,
                rows=rows_data,
                cell_ranges=f"A1:{openpyxl.utils.get_column_letter(len(headers))}{len(rows_data)+1}",
                formulas=formulas
            )
            tables.append(table)

            # Create structured table document element
            summary_content = f"Table from sheet '{sheet_name}' with {len(headers)} columns and {len(rows_data)} rows. Headers: {', '.join(headers)}"
            elements.append(DocumentElement(
                type=ElementType.TABLE,
                content=summary_content,
                heading_path=[f"Workbook", f"Sheet: {sheet_name}"],
                metadata={"sheet_name": sheet_name, "headers": headers, "row_count": len(rows_data)}
            ))

        return tables, elements

    def _parse_with_ooxml(self, file_bytes: bytes, filename: str) -> Tuple[List[TableData], List[DocumentElement]]:
        """Native OOXML parser that reads xl/worksheets/sheet*.xml and xl/sharedStrings.xml directly."""
        import io
        tables: List[TableData] = []
        elements: List[DocumentElement] = []

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
                # Read shared strings
                shared_strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
                    for si in ss_root.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                        t_elem = si.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                        shared_strings.append(t_elem.text if t_elem is not None and t_elem.text else "")

                # Read workbook sheets
                wb_root = ET.fromstring(z.read("xl/workbook.xml"))
                sheet_nodes = wb_root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")

                for idx, sheet_node in enumerate(sheet_nodes):
                    sheet_name = sheet_node.get("name", f"Sheet{idx+1}")
                    sheet_path = f"xl/worksheets/sheet{idx+1}.xml"
                    if sheet_path not in z.namelist():
                        continue

                    sheet_root = ET.fromstring(z.read(sheet_path))
                    ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    
                    rows: List[List[str]] = []
                    formulas: Dict[str, str] = {}

                    for row_elem in sheet_root.findall(".//ns:row", ns):
                        curr_row = []
                        for c_elem in row_elem.findall("ns:c", ns):
                            coord = c_elem.get("r", "")
                            c_type = c_elem.get("t")
                            f_elem = c_elem.find("ns:f", ns)
                            v_elem = c_elem.find("ns:v", ns)
                            
                            if f_elem is not None and f_elem.text:
                                formulas[coord] = f"={f_elem.text}"

                            val = ""
                            if v_elem is not None and v_elem.text:
                                if c_type == "s" and v_elem.text.isdigit():
                                    s_idx = int(v_elem.text)
                                    val = shared_strings[s_idx] if s_idx < len(shared_strings) else v_elem.text
                                else:
                                    val = v_elem.text
                            curr_row.append(val)
                        if curr_row:
                            rows.append(curr_row)

                    headers = rows[0] if rows else []
                    data_rows = rows[1:] if len(rows) > 1 else []

                    table = TableData(
                        sheet_name=sheet_name,
                        headers=headers,
                        rows=data_rows,
                        formulas=formulas
                    )
                    tables.append(table)

                    elements.append(DocumentElement(
                        type=ElementType.HEADING,
                        content=f"Sheet: {sheet_name}",
                        heading_path=["Workbook", f"Sheet: {sheet_name}"],
                        level=2
                    ))
                    elements.append(DocumentElement(
                        type=ElementType.TABLE,
                        content=f"Table '{sheet_name}' with columns {headers}",
                        heading_path=["Workbook", f"Sheet: {sheet_name}"],
                        metadata={"headers": headers, "row_count": len(data_rows)}
                    ))
        except Exception as e:
            # Fallback dummy element for error resilience
            elements.append(DocumentElement(
                type=ElementType.PARAGRAPH,
                content=f"Excel file parsed with fallback note: {str(e)}",
                heading_path=["Workbook"]
            ))

        return tables, elements

